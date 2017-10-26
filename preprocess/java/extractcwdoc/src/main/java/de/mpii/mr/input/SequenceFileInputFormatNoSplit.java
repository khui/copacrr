package de.mpii.mr.input;

import java.io.IOException;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileRecordReader;

/**
 *
 * @author khui
 * @param <K>
 * @param <V>
 */
public class SequenceFileInputFormatNoSplit<K, V> extends SequenceFileInputFormat<K, V> {

    @Override
    public RecordReader<K, V> createRecordReader(InputSplit split,
            TaskAttemptContext context
    ) throws IOException {
        return new SequenceFileRecordReader<>();
    }

    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

}
