package com.karthik.transformer.core;

/**
 * Dense 2D matrix used by every transformer operation in this project.
 *
 * Embeddings, attention scores, and linear projections are all matrix ops.
 * Methods return new instances (no in-place mutation) so call sites stay easy to reason about.
 *
 * @author Karthik Goud
 */
public class Matrix {

    private final double[][] data;
    public final int rows;
    public final int cols;

    // ─── Constructors ────────────────────────────────────────────────────────

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = deepCopy(data);
    }

    /** Factory: identity matrix (diagonal = 1) */
    public static Matrix identity(int size) {
        Matrix m = new Matrix(size, size);
        for (int i = 0; i < size; i++) m.set(i, i, 1.0);
        return m;
    }

    /** Factory: matrix filled with random small values (like weight init) */
    public static Matrix random(int rows, int cols, double scale) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, (Math.random() * 2 - 1) * scale);
        return m;
    }

    /** Factory: single-row matrix from a 1D array (a vector) */
    public static Matrix rowVector(double[] values) {
        Matrix m = new Matrix(1, values.length);
        for (int j = 0; j < values.length; j++) m.set(0, j, values[j]);
        return m;
    }

    /** Factory: single-column matrix from a 1D array */
    public static Matrix colVector(double[] values) {
        Matrix m = new Matrix(values.length, 1);
        for (int i = 0; i < values.length; i++) m.set(i, 0, values[i]);
        return m;
    }

    // ─── Core Operations ─────────────────────────────────────────────────────

    /**
     * Matrix multiplication — the single most important operation in deep learning.
     *
     * Think of it as: for each output cell (i,j), take the dot product of
     * row i from A and column j from B. Every attention score, every embedding
     * lookup, every linear projection is this operation.
     *
     * Requires: this.cols == other.rows
     */
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch: (%d×%d) × (%d×%d) — inner dims must match",
                    this.rows, this.cols, other.rows, other.cols));
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < other.cols; j++)
                for (int k = 0; k < this.cols; k++)
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
        return result;
    }

    /**
     * Transpose — flip rows and columns.
     *
     * In attention: Q × K^T means we transpose K so the shapes align.
     * Visually: a (3×4) matrix becomes (4×3).
     */
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < this.cols; j++)
                result.data[j][i] = this.data[i][j];
        return result;
    }

    /**
     * Element-wise addition — add two matrices of the same shape.
     * Used in residual connections: output = layer(x) + x
     */
    public Matrix add(Matrix other) {
        checkSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = this.data[i][j] + other.data[i][j];
        return result;
    }

    /**
     * Element-wise subtraction.
     */
    public Matrix subtract(Matrix other) {
        checkSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = this.data[i][j] - other.data[i][j];
        return result;
    }

    /**
     * Scale every element by a scalar.
     * Used in attention: divide scores by sqrt(d_k) to prevent vanishing gradients.
     */
    public Matrix scale(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = this.data[i][j] * scalar;
        return result;
    }

    /**
     * Dot product of two vectors (1D, treated as flat arrays).
     *
     * Think of it as a "similarity score" — the higher the dot product,
     * the more aligned two vectors are. This is how attention decides
     * which tokens are relevant to each other.
     */
    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Vectors must be same length for dot product");
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /**
     * L2 norm (magnitude) of a vector — sqrt(sum of squares).
     * Used to normalize vectors before cosine similarity.
     */
    public static double norm(double[] vec) {
        double sumSq = 0;
        for (double v : vec) sumSq += v * v;
        return Math.sqrt(sumSq);
    }

    /**
     * Cosine similarity between two vectors.
     * Range: -1 (opposite) to +1 (identical direction).
     * Used to compare embeddings — how semantically similar are two tokens?
     */
    public static double cosineSimilarity(double[] a, double[] b) {
        return dotProduct(a, b) / (norm(a) * norm(b) + 1e-8);
    }

    /**
     * Apply ReLU activation: max(0, x)
     * Used in feed-forward layers inside transformers.
     */
    public Matrix relu() {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = Math.max(0, this.data[i][j]);
        return result;
    }

    /**
     * Row-wise softmax — converts raw scores into a probability distribution.
     *
     * The trick: subtract max before exp() to prevent numerical overflow.
     * Used in attention to turn raw attention scores into weights that sum to 1.
     */
    public Matrix softmax() {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            // Step 1: find max in row (numerical stability)
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++)
                if (this.data[i][j] > max) max = this.data[i][j];

            // Step 2: exp(x - max)
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = Math.exp(this.data[i][j] - max);
                sum += result.data[i][j];
            }

            // Step 3: divide by sum so row sums to 1
            for (int j = 0; j < cols; j++)
                result.data[i][j] /= sum;
        }
        return result;
    }

    /**
     * Layer normalization — normalizes each row to mean=0, std=1, then scales.
     * Keeps training stable. Applied after attention and feed-forward in transformers.
     */
    public Matrix layerNorm(double epsilon) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            // compute mean
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += this.data[i][j];
            mean /= cols;

            // compute variance
            double variance = 0;
            for (int j = 0; j < cols; j++) {
                double diff = this.data[i][j] - mean;
                variance += diff * diff;
            }
            variance /= cols;

            // normalize
            double std = Math.sqrt(variance + epsilon);
            for (int j = 0; j < cols; j++)
                result.data[i][j] = (this.data[i][j] - mean) / std;
        }
        return result;
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    public double get(int row, int col) { return data[row][col]; }
    public void set(int row, int col, double val) { data[row][col] = val; }

    public double[] getRow(int row) {
        double[] r = new double[cols];
        System.arraycopy(data[row], 0, r, 0, cols);
        return r;
    }

    public double[] getCol(int col) {
        double[] c = new double[rows];
        for (int i = 0; i < rows; i++) c[i] = data[i][col];
        return c;
    }

    public double[][] getData() { return deepCopy(data); }

    // ─── Display ─────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Matrix (%d×%d):\n", rows, cols));
        for (int i = 0; i < rows; i++) {
            sb.append("  [ ");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%7.4f", data[i][j]));
                if (j < cols - 1) sb.append(", ");
            }
            sb.append(" ]\n");
        }
        return sb.toString();
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private void checkSameShape(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols)
            throw new IllegalArgumentException(
                String.format("Shape mismatch: (%d×%d) vs (%d×%d)",
                    this.rows, this.cols, other.rows, other.cols));
    }

    private static double[][] deepCopy(double[][] src) {
        double[][] copy = new double[src.length][src[0].length];
        for (int i = 0; i < src.length; i++)
            System.arraycopy(src[i], 0, copy[i], 0, src[i].length);
        return copy;
    }
}
