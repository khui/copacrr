package de.mpii.mr.tasks;

import de.mpii.mr.input.clueweb.FilteredCw09InputFormat;
import de.mpii.mr.input.clueweb.ClueWebWarcRecord;
import de.mpii.mr.input.clueweb.FilteredCw12InputFormat;
import de.mpii.GlobalConstants;

import java.util.Iterator;

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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/**
 *
 * @author khui
 */
public class ExtractCwidUrl extends Configured implements Tool {
    
    private static final Logger logger = Logger.getLogger(ExtractCwidUrl.class);
    
    private static class Constants {
        
        static String SEP = "/";
        
    }
    
    private static enum Counter {
        
        TOTAL, PAGES, ERRORS, SKIPPEDPAGES, SKIPPEDTERMS, DFFILTERED
    };
    
    @Override
    public int run(String[] args) throws Exception {
        // command-line parsing
        Options options = new Options();
        options.addOption("r", "root", true, "root directory");
        options.addOption("c", "collection", true, "collection directory");
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
        String outputdir = cmd.getOptionValue('r');
        // "cw09 or cw12"
        String colname = cmd.getOptionValue('n');
        String log4jconf = cmd.getOptionValue('l');
        org.apache.log4j.PropertyConfigurator.configure(log4jconf);
        LogManager.getRootLogger().setLevel(Level.INFO);

        /**
         * start setting up job
         */
        Job job = new Job(getConf(), ExtractCwidUrl.class.getSimpleName() + " Fetch url for given cwid: " + colname);
        job.setJarByClass(ExtractCwidUrl.class);
        job.setNumReduceTasks(1);
        //job.setUserClassesTakesPrecedence(true);

        DistributedCache.addCacheFile(new Path("/user/khui/similarity/cwids/cwids." + colname).toUri(), job.getConfiguration());
        job.getConfiguration().set(GlobalConstants.CWIDS_FILE_NAME_PREFIX, "cwids." + colname);
        Path[] inputPaths = DFSUtils.recurse(inputcols, "warc.gz", job.getConfiguration());
        Path outputpath = new Path(outputdir + Constants.SEP + "cwid-url-" + colname);
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
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        job.setMapperClass(Cwid2Url.class);
        job.setReducerClass(ConcatenateLines.class);
        
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        logger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
        return 0;
    }
    
    
    
    
    
    private static class Cwid2Url extends Mapper<Text, ClueWebWarcRecord, Text, Text> {
        
        private final Text url = new Text();
        
        @Override
        public void map(Text key, ClueWebWarcRecord doc, Context context) {
            String docid = key.toString();
            if (docid != null) {
                try {
                    context.getCounter(Counter.PAGES).increment(1);
                    String urlstr = doc.getHeaderMetadataItem("WARC-Target-URI");
                    url.set(urlstr);
                    if (urlstr != null) {
                        context.write(key, url);
                    } else {
                        logger.error("urlstr is null in mapper");
                    }
                    
                } catch (Exception ex) {
                    context.getCounter(Counter.ERRORS).increment(1);
                    logger.error("", ex);
                }
                
            } else {
                logger.error("docid is " + docid);
            }
        }
        
    }
    
    private static class ConcatenateLines extends Reducer<Text, Text, Text, Text> {
        
        @Override
        public void reduce(Text docid, Iterable<Text> lines, Context context) {
            try {
                Iterator<Text> lineiterator = lines.iterator();
                while (lineiterator.hasNext()) {
                    context.write(docid, lineiterator.next());
                }
            } catch (Exception ex) {
                context.getCounter(Counter.ERRORS).increment(1);
                logger.error("", ex);
            }
        }
        
    }
    
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ExtractCwidUrl(), args);
        System.exit(exitCode);
    }
    
}
