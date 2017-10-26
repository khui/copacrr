package de.mpii.mr.input.clueweb;

import de.mpii.GlobalConstants;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.log4j.Logger;

/**
 *
 * @author khui
 */
public class FilteredCw09InputFormat extends ClueWeb09InputFormat {

    private static final Logger logger = Logger.getLogger(FilteredCw09InputFormat.class);

    private final Map<String, TIntList> cwidsQids;

    public FilteredCw09InputFormat() {
        this.cwidsQids = new HashMap<>();
    }

    @Override
    public RecordReader<Text, ClueWeb09WarcRecord> createRecordReader(InputSplit is, TaskAttemptContext tac) throws IOException, InterruptedException {
        return new CW09FilteredRecordReader(is, tac);
    }
    /**
     * the cwid list could be in two types of formats
     * 1) each line contains a cwid
     * 2) each line contains cwid" "qid
     */
    public class CW09FilteredRecordReader extends ClueWeb09InputFormat.ClueWeb09RecordReader {

        public CW09FilteredRecordReader(InputSplit is, TaskAttemptContext tac) throws IOException {

            Configuration conf = tac.getConfiguration();
            String cwidfilename = conf.get(GlobalConstants.CWIDS_FILE_NAME_PREFIX);
            Path[] paths = DistributedCache.getLocalCacheFiles(conf);
            FileSystem fs = FileSystem.getLocal(conf);
            for (Path path : paths) {
                if (path.getName().equals(cwidfilename)) {
                    if (fs.exists(path)) {
                        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
                        while (br.ready()) {
                            String cwid;
                            String line = br.readLine().trim();
                            String[] cols = line.split(" ");
                            if (cols.length == 1) {
                                if (line.length() > 1) {
                                    cwidsQids.put(line, null);
                                }
                            } else if (cols.length == 2) {
                                int qid = Integer.parseInt(cols[1]);
                                if (qid > 200){
                                    continue;
                                }
                                cwid = cols[0];

                                if (cwid.length() > 1) {
                                    if (!cwidsQids.containsKey(cwid)) {
                                        cwidsQids.put(cwid, new TIntArrayList());
                                    }
                                    cwidsQids.get(cwid).add(qid);
                                }
                            }

                        }
                        logger.info("FINISH read in " + path.getName() + " : " + cwidsQids.size());
                        br.close();
                    } else {
                        logger.error(path.getName() + " does not exists!");
                    }
                }
            }
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            while (true) {
                try {
                    value = ClueWeb09WarcRecord.readNextWarcRecord(in);
                } catch (Exception ex) {
                    logger.error("", ex);
                    value = null;
                }
                if (value == null) {
                    return false;
                }
                String docid = value.getDocid();
                if (docid != null) {
                    if (cwidsQids.containsKey(docid)) {
                        if (cwidsQids.get(docid) == null) {
                            key = new Text(docid);
                        } else {
                            //qid-qid-qid--1
                            StringBuilder qids = new StringBuilder();
                            for (int qid : cwidsQids.get(docid).toArray()) {
                                qids.append(qid).append("-");
                            }
                            qids.append(0);
                            String qidstr = qids.toString();
                            String keystr = qidstr + "." + docid;
                            key = new Text(keystr);
                        }
                        pos = filePosition.getPos();
                        return true;
                    }
                } else {
                    logger.error("docid is null");
                }
            }
        }
    }

}
