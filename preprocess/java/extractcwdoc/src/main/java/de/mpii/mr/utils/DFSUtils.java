package de.mpii.mr.utils;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;

/**
 *
 * @author Klaus Bererich
 */
public class DFSUtils {

    private final static Logger logger = Logger.getLogger(DFSUtils.class.getName());

    public static Path[] recurse(String input, String suffix, Configuration conf) throws IOException {
        List<Path> result = new LinkedList<>();
        FileSystem fs = FileSystem.get(conf);
        String[] paths = input.split(":");
        for (String path : paths) {
            recurse(new Path(path), fs, suffix, result);
        }
        return result.toArray(new Path[0]);
    }

    private static void recurse(Path path, FileSystem fs, String suffix, List<Path> paths) throws IOException {
        if (fs.getFileStatus(path).isDir()) {
            for (FileStatus status : fs.listStatus(path)) {
                recurse(status.getPath(), fs, suffix, paths);
            }
        } else {
            if (path.getName().endsWith(suffix)) {
                paths.add(path);
            }
        }
    }

}
