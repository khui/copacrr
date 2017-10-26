package de.mpii.mr.input.clueweb;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.apache.hadoop.io.Writable;

/**
 *
 * @author khui
 */
public interface ClueWebWarcRecord extends Writable,  java.io.Serializable   {

    String getContent();

    String getDocid();

    // WARC-Target-URI
    String getHeaderMetadataItem(String key);

    Set<Map.Entry<String, String>> getHeaderMetadata();
    
    @Override
    public void write(DataOutput out) throws IOException;
    
    @Override
    public void readFields(DataInput in) throws IOException;

}
