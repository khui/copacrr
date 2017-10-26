package de.mpii.mr.input.clueweb;

import java.io.DataInputStream;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.Seekable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.log4j.Logger;

/**
 *
 * @author khui
 */
public class ClueWeb12InputFormat extends FileInputFormat<Text, ClueWeb12WarcRecord> {

    private static final Logger logger = Logger.getLogger(ClueWeb12InputFormat.class);

    @Override
    public RecordReader<Text, ClueWeb12WarcRecord> createRecordReader(InputSplit is, TaskAttemptContext tac) throws IOException, InterruptedException {
        return new ClueWeb12RecordReader();
    }

    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

    public class ClueWeb12RecordReader extends RecordReader<Text, ClueWeb12WarcRecord> {

        protected long start;
        protected long pos;
        protected long end;
        protected Seekable filePosition;
        protected DataInputStream in;
        protected Text key;
        protected ClueWeb12WarcRecord value;

        @Override
        public void initialize(InputSplit is, TaskAttemptContext tac) throws IOException, InterruptedException {
            FileSplit split = (FileSplit) is;
            Configuration conf = tac.getConfiguration();
            start = split.getStart();
            pos = start;
            end = start + split.getLength();
            Path path = split.getPath();
            CompressionCodecFactory compressionCodecs = new CompressionCodecFactory(conf);
            CompressionCodec codec = compressionCodecs.getCodec(path);
            FileSystem fs = path.getFileSystem(conf);
            FSDataInputStream fileIn = fs.open(split.getPath());
            in = new DataInputStream(codec.createInputStream(fileIn, codec.createDecompressor()));
            filePosition = fileIn;
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            while (true) {
                try {
                    value = ClueWeb12WarcRecord.readNextWarcRecord(in);
                } catch (Exception ex) {
                    logger.error("", ex);
                    value = null;
                }
                if (value == null) {
                    return false;
                }
                String docid = value.getDocid();
                if (docid != null) {
                    key = new Text(docid);
                    pos = filePosition.getPos();
                    return true;
                } else {
                    logger.error("docid is null");
                }
            }
        }

        @Override
        public Text getCurrentKey() throws IOException, InterruptedException {
            return key;
        }

        @Override
        public ClueWeb12WarcRecord getCurrentValue() throws IOException, InterruptedException {
            return value;
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            if (start == end) {
                return 0.0f;
            } else {
                return Math.min(1.0f, (pos - start) / (float) (end - start));
            }
        }

        @Override
        public void close() throws IOException {
            if (in != null) {
                in.close();
            }
        }

    }

}
