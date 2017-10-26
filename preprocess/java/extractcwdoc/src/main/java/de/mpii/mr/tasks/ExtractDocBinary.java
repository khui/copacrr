package de.mpii.mr.tasks;

import de.mpii.GlobalConstants;
import de.mpii.mr.input.clueweb.FilteredCw09InputFormat;
import de.mpii.mr.input.clueweb.ClueWebWarcRecord;
import de.mpii.mr.input.clueweb.FilteredCw12InputFormat;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import de.mpii.mr.utils.DFSUtils;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/**
 *
 * @author khui
 */
public class ExtractDocBinary extends Configured implements Tool {

    private static final Logger logger = Logger.getLogger(ExtractDocBinary.class);

    private static class Constants {

        static String SEP = "/";

    }

    private static enum Counter {

        PAGES, ERRORS, QIDPAGES
    }

    @Override
    public int run(String[] args) throws Exception {
        // command-line parsing
        Options options = new Options();
        options.addOption("o", "root", true, "root directory");
        options.addOption("c", "collection", true, "collection directory");
        options.addOption("f", "filterfile", true, "cwid qid file list for filtering the docs");
        options.addOption("l", "log4jconf", true, "log4jconf");
        options.addOption("n", "dataname", true, "name of collection to deal with");
        options.addOption("h", "help", false, "print this message");

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = parser.parse(options, args);

        if (cmd.hasOption("h") || cmd.getOptions().length == 0) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(this.getClass().toString(), options);
            System.exit(0);
        }

        String inputcols = cmd.getOptionValue('c');
        String filterlistfile = cmd.getOptionValue('f');
        String outputdir = cmd.getOptionValue('o');
        // "cw09 or cw12"
        String colname = cmd.getOptionValue('n');
        if (cmd.hasOption('l')) {
            String log4jconf = cmd.getOptionValue('l');
            org.apache.log4j.PropertyConfigurator.configure(log4jconf);
        }
        LogManager.getRootLogger().setLevel(Level.INFO);

        /**
         * start setting up job
         */
        Job job = new Job(getConf(), ExtractDocBinary.class.getSimpleName() + " Extract doc for qrels: " + colname);
        job.setJarByClass(ExtractDocBinary.class);
        job.setNumReduceTasks(16);
        //job.setUserClassesTakesPrecedence(true);

        DistributedCache.addCacheFile(new Path(filterlistfile).toUri(), job.getConfiguration());
        job.getConfiguration().set(GlobalConstants.CWIDS_FILE_NAME_PREFIX, new Path(filterlistfile).getName());
        Path[] inputPaths = DFSUtils.recurse(inputcols, "warc.gz", job.getConfiguration());
        Path outputpath = new Path(outputdir + Constants.SEP + colname);
        if (FileSystem.get(getConf()).exists(outputpath)) {
            logger.warn(outputpath.getName() + " is being removed");
            FileSystem.get(getConf()).delete(outputpath, true);
        }

        FileInputFormat.setInputPaths(job, inputPaths);
        FileOutputFormat.setOutputPath(job, outputpath);
        switch (colname) {
            case "cw09":
                job.setInputFormatClass(FilteredCw09InputFormat.class);
                break;
            case "cw12":
                job.setInputFormatClass(FilteredCw12InputFormat.class);
                break;
            default:
                logger.error(colname + " is unsupported data name");
                System.exit(1);
        }

        LazyOutputFormat.setOutputFormatClass(job, SequenceFileOutputFormat.class);
        //FileOutputFormat.setCompressOutput(job, true);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(BytesWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(BytesWritable.class);

        for (int qid = 1; qid <= 300; qid++) {
            MultipleOutputs.addNamedOutput(job, String.valueOf(qid), SequenceFileOutputFormat.class, IntWritable.class, BytesWritable.class);
        }

        job.setMapperClass(Cwrecord2ByteArray.class);
        job.setReducerClass(MergePerQuery.class);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        logger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
        return 0;
    }

    private static class Cwrecord2ByteArray extends Mapper<Text, ClueWebWarcRecord, IntWritable, BytesWritable> {

        private final ByteArrayOutputStream byteOut = new ByteArrayOutputStream();

        private final BytesWritable bytesWritable = new BytesWritable();

        private final IntWritable qidWritable = new IntWritable();

        private byte[] buffer;

        @Override
        public void map(Text key, ClueWebWarcRecord doc, Context context) throws IOException {
            String line = key.toString();
            String[] cols = line.split("\\.");
            String docid = cols[1];
            String[] qids = cols[0].split("-");
            if (docid != null) {
                try {
                    context.getCounter(Counter.PAGES).increment(1);
                    doc.write(new DataOutputStream(byteOut));
                    buffer = byteOut.toByteArray();
                    byteOut.reset();
                    bytesWritable.set(buffer, 0, buffer.length);
                    for (String qidStr : qids) {
                        int qid = Integer.parseInt(qidStr);
                        if (qid > 0) {
                            qidWritable.set(qid);
                            context.getCounter(Counter.QIDPAGES).increment(1);
                            context.write(qidWritable, bytesWritable);
                        }
                    }
                } catch (Exception ex) {
                    context.getCounter(Counter.ERRORS).increment(1);
                    logger.error("", ex);
                }
            } else {
                logger.error("docid is " + docid);
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            byteOut.close();
        }

    }

    private static class MergePerQuery extends Reducer<IntWritable, BytesWritable, IntWritable, BytesWritable> {

        @Override
        public void reduce(IntWritable key, Iterable<BytesWritable> values, Context context) throws IOException, InterruptedException {
            MultipleOutputs mos = new MultipleOutputs(context);
            int qid = key.get();
            for (BytesWritable doc : values) {
                mos.write(String.valueOf(qid), key, doc);
            }
            logger.info(qid + " finished");
            mos.close();
        }

    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ExtractDocBinary(), args);
        System.exit(exitCode);
    }

}
