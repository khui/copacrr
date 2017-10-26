package de.mpii;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import java.util.List;
import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * java implementation of our maxrep algorithm. given n data points, we target
 * at selecting k points that can best represent the remaining data points in
 * terms of similarity among data points
 *
 * @author khui
 */
public class MaxRepSelection {

    static Logger logger = Logger.getLogger(MaxRepSelection.class.getName());

    private List<INDArray> datapoints;

    private final double[] datapointWeights;

    private final int n;

    private final double[][] similarityMatrix;

    private final double[] maxRepInSelectedsetVector;
    // the selected data point, representing by the index of the data in the input data points array
    private final TIntList selectedL = new TIntArrayList();

    private final double similarityThreshold;

    public MaxRepSelection(List<INDArray> datapoint, double similarityThreshold) {
        this.datapoints = datapoint;
        this.n = datapoints.size();
        this.similarityThreshold = similarityThreshold;
        this.datapointWeights = new double[datapoint.size()];
        updateNodeWeight();
        this.maxRepInSelectedsetVector = new double[n];
        this.similarityMatrix = new double[n][];
        for (int i = 0; i < n; i++) {
            this.similarityMatrix[i] = new double[n];
        }
        initSimimatrix();
    }

    public MaxRepSelection(double[][] similarityMatrix, double similarityThreshold) {
        this.n = similarityMatrix.length;
        this.similarityThreshold = similarityThreshold;
        this.datapointWeights = new double[this.n];
        updateNodeWeight();
        this.maxRepInSelectedsetVector = new double[n];
        this.similarityMatrix = similarityMatrix;
    }



    private void updateNodeWeight() {
        for (int i = 0; i < datapointWeights.length; i++) {
            datapointWeights[i] = 1;
        }
    }

    /**
     * main entrance of the method, pick up k data points as the representative
     * points in terms of weighted similarity. return the keys of the selected
     * centroids.
     *
     * @param k
     * @return
     */
    public int[] selectMaxRep(int k) {
        while (selectedL.size() < Math.min(n, k)) {
            int max_simigain_index = getTopRep();
            if (max_simigain_index >= 0) {
                selectedL.add(max_simigain_index);
                updateMaxRepVector(max_simigain_index);
            } else {
                logger.error("max_ind is negative: " + max_simigain_index + " " + selectedL.size());
                break;
            }
        }
        return selectedL.toArray();
    }

    /**
     * in each round, we pick up the data point that can provide the maximum
     * similarity gain for the selected points set w.r.t. the complete data set.
     * for each data point i among all points, the comparison between its
     * weighted similarity against every other data points j, and j'th maximum
     * similarity w.r.t. all selected points in L, aggregating the positive gain
     * in delta_simi_sum. Afterward, pick up the data point i with max
     * delta_simi_sum to add to the selected set. the weight for each data point
     * indicates their importance. intuitively, we want to pick up the data
     * point into the selected set that can maximum increase the selected set's
     * weighted representativeness in terms of similarity.
     *
     * @return
     */
    private int getTopRep() {
        int max_ind = -1;
        double max_delta = 0;
        // go thru all data points as candidate data points to be selected
        for (int i = 0; i < n; i++) {
            double delta_simi_sum = 0;
            // for i-th data point, compare it with every other data points
            // to compute the possible similarity gain, aggregating in delta_simi_sum
            for (int j = 0; j < n; j++) {
                double ij_similarity = similarityMatrix[i][j] * datapointWeights[j];
                double max_simi_j = maxRepInSelectedsetVector[j];
                delta_simi_sum += Math.max(ij_similarity - max_simi_j, 0);
            }

            if (delta_simi_sum > max_delta) {
                max_ind = i;
                max_delta = delta_simi_sum;
            }
        }
        return max_ind;
    }

    /**
     * update the max-similarity array, representing the maximum similarity
     * among the selected data points and current data point. the similarity
     * being updated is the multiplication between similarity and the weight of
     * current data point.
     *
     * @param selected_data_index
     */
    private void updateMaxRepVector(int selected_data_index) {
        for (int i = 0; i < n; i++) {
            maxRepInSelectedsetVector[i] = Math.max(maxRepInSelectedsetVector[i],
                    similarityMatrix[i][selected_data_index] * datapointWeights[i]);
        }
    }

    /**
     * generate n X n similarity matrix for the input n data points, similarity
     * value should lay between 0 and 1, inclusive
     */
    private void initSimimatrix() {
        // compute similarity for upper diangel  
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {

                double similarity = Transforms.cosineSim(datapoints.get(i), datapoints.get(j));
                        //cosineSimilarity(datapoints.get(i), datapoints.get(j));
                if (similarity < 0 || similarity > 1) {
                    similarity = (similarity < 0 ? 0 : 1);
                }
                // regard datapoints far away enough as non-similar
                if (similarity < similarityThreshold) {
                    similarity = 0;
                }
                this.similarityMatrix[i][j] = similarity;
            }
        }
        // make the similarity matrix symmetrics in favor of further computation
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    this.similarityMatrix[i][j] = 1;
                } else {
                    this.similarityMatrix[i][j] = this.similarityMatrix[j][i];
                }
            }
        }
    }

    private double cosineSimilarity(float[] vectorA, float[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

}
