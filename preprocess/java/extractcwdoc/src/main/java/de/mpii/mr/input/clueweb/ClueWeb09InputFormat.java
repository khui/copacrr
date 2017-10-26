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
public class ClueWeb09InputFormat extends FileInputFormat<Text, ClueWeb09WarcRecord> {

    private static final Logger logger = Logger.getLogger(ClueWeb09InputFormat.class);

    @Override
    public RecordReader<Text, ClueWeb09WarcRecord> createRecordReader(InputSplit is, TaskAttemptContext tac) throws IOException, InterruptedException {
        return new ClueWeb09RecordReader();
    }

    @Override
    public boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

    public class ClueWeb09RecordReader extends RecordReader<Text, ClueWeb09WarcRecord> {

        protected long start;
        protected long pos;
        protected long end;
        protected DataInputStream in;
        protected Text key;
        protected ClueWeb09WarcRecord value;
        protected Seekable filePosition;

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
                value = ClueWeb09WarcRecord.readNextWarcRecord(in);
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
        public ClueWeb09WarcRecord getCurrentValue() throws IOException, InterruptedException {
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
