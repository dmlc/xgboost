/*
 Copyright (c) 2014-2025 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test cases for LabeledPoint class, covering both dense and sparse vector scenarios.
 */
public class LabeledPointTest {

    @Test
    public void testDenseVectorConstructor() {
        // Test dense vector (indices = null)
        float label = 1.5f;
        int size = 4;
        int[] indices = null; // null indicates dense vector
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        float weight = 2.0f;
        int group = 5;
        float baseMargin = 0.5f;

        LabeledPoint point = new LabeledPoint(label, size, indices, values, weight, group, baseMargin);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertNull(point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(weight, point.weight(), 0.001f);
        assertEquals(group, point.group());
        assertEquals(baseMargin, point.baseMargin(), 0.001f);
    }

    @Test
    public void testSparseVectorConstructor() {
        // Test sparse vector (indices != null)
        float label = -1.0f;
        int size = 10;
        int[] indices = {0, 3, 7, 9}; // sparse indices
        float[] values = {1.5f, -2.0f, 3.5f, 0.8f};
        float weight = 1.5f;
        int group = 2;
        float baseMargin = -0.2f;

        LabeledPoint point = new LabeledPoint(label, size, indices, values, weight, group, baseMargin);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(weight, point.weight(), 0.001f);
        assertEquals(group, point.group());
        assertEquals(baseMargin, point.baseMargin(), 0.001f);
    }

    @Test
    public void testSimpleConstructor() {
        // Test constructor with only label, size, indices, and values
        float label = 0.0f;
        int size = 3;
        int[] indices = {0, 2};
        float[] values = {1.0f, -1.0f};

        LabeledPoint point = new LabeledPoint(label, size, indices, values);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(1.0f, point.weight(), 0.001f); // default weight
        assertEquals(-1, point.group()); // default group
        assertTrue(Float.isNaN(point.baseMargin())); // default baseMargin
    }

    @Test
    public void testConstructorWithWeight() {
        // Test constructor with weight
        float label = 2.5f;
        int size = 5;
        int[] indices = null; // dense
        float[] values = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f};
        float weight = 3.0f;

        LabeledPoint point = new LabeledPoint(label, size, indices, values, weight);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertNull(point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(weight, point.weight(), 0.001f);
        assertEquals(-1, point.group()); // default group
        assertTrue(Float.isNaN(point.baseMargin())); // default baseMargin
    }

    @Test
    public void testConstructorWithWeightAndGroup() {
        // Test constructor with weight and group
        float label = -2.0f;
        int size = 6;
        int[] indices = {1, 3, 5};
        float[] values = {2.0f, -1.0f, 4.0f};
        float weight = 0.5f;
        int group = 10;

        LabeledPoint point = new LabeledPoint(label, size, indices, values, weight, group);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(weight, point.weight(), 0.001f);
        assertEquals(group, point.group());
        assertTrue(Float.isNaN(point.baseMargin())); // default baseMargin
    }

    @Test
    public void testDenseVectorWithZeros() {
        // Test dense vector with zero values
        float label = 1.0f;
        int size = 5;
        int[] indices = null;
        float[] values = {0.0f, 1.0f, 0.0f, 0.0f, 2.0f};

        LabeledPoint point = new LabeledPoint(label, size, indices, values);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertNull(point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
    }

    @Test
    public void testSparseVectorEmpty() {
        // Test sparse vector with no non-zero elements
        float label = 0.5f;
        int size = 100;
        int[] indices = {};
        float[] values = {};

        LabeledPoint point = new LabeledPoint(label, size, indices, values);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
        assertEquals(0, point.indices().length);
        assertEquals(0, point.values().length);
    }

    @Test
    public void testEqualsDenseVsSparse() {
        // Test that dense and sparse representations are not equal even if they represent the same data
        float label = 1.0f;
        
        // Dense representation
        int denseSize = 4;
        int[] denseIndices = null;
        float[] denseValues = {1.0f, 0.0f, 2.0f, 0.0f};
        
        // Sparse representation (same data)
        int sparseSize = 4;
        int[] sparseIndices = {0, 2};
        float[] sparseValues = {1.0f, 2.0f};

        LabeledPoint densePoint = new LabeledPoint(label, denseSize, denseIndices, denseValues);
        LabeledPoint sparsePoint = new LabeledPoint(label, sparseSize, sparseIndices, sparseValues);

        assertNotEquals(densePoint, sparsePoint);
    }

    @Test
    public void testSpecialFloatValues() {
        // Test with special float values (NaN, infinity)
        float label = Float.POSITIVE_INFINITY;
        int size = 3;
        int[] indices = {0, 1, 2};
        float[] values = {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY};
        float weight = Float.NaN;
        int group = 0;
        float baseMargin = Float.NEGATIVE_INFINITY;

        LabeledPoint point = new LabeledPoint(label, size, indices, values, weight, group, baseMargin);

        assertEquals(Float.POSITIVE_INFINITY, point.label(), 0.0f);
        assertTrue(Float.isNaN(point.values()[0]));
        assertEquals(Float.NEGATIVE_INFINITY, point.values()[1], 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, point.values()[2], 0.0f);
        assertTrue(Float.isNaN(point.weight()));
        assertEquals(Float.NEGATIVE_INFINITY, point.baseMargin(), 0.0f);
    }

    @Test
    public void testLargeSparseDimension() {
        // Test sparse vector with large dimension but few non-zero elements
        float label = 0.8f;
        int size = 1000000; // 1 million dimensions
        int[] indices = {0, 500000, 999999}; // only 3 non-zero elements
        float[] values = {1.0f, 2.0f, 3.0f};

        LabeledPoint point = new LabeledPoint(label, size, indices, values);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertEquals(3, point.indices().length);
        assertEquals(3, point.values().length);
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
    }

    @Test(expected = AssertionError.class)
    public void testInvalidSparseIndices() {
        // Test assertion failure when size is less than indices length
        float label = 1.0f;
        int size = 2; // size is 2
        int[] indices = {0, 1, 2}; // but we have 3 indices (invalid)
        float[] values = {1.0f, 2.0f, 3.0f};

        new LabeledPoint(label, size, indices, values);
    }

    @Test
    public void testBoundaryIndices() {
        // Test sparse vector with indices at boundaries
        float label = -0.5f;
        int size = 10;
        int[] indices = {0, 9}; // first and last indices
        float[] values = {-1.0f, 1.0f};

        LabeledPoint point = new LabeledPoint(label, size, indices, values);

        assertEquals(label, point.label(), 0.001f);
        assertEquals(size, point.size());
        assertArrayEquals(indices, point.indices());
        assertArrayEquals(values, point.values(), 0.001f);
    }

    @Test
    public void testSingleElementVectors() {
        // Test single element dense vector
        float label = 5.0f;
        int size = 1;
        int[] indices = null;
        float[] values = {42.0f};

        LabeledPoint densePoint = new LabeledPoint(label, size, indices, values);
        assertEquals(1, densePoint.values().length);
        assertEquals(42.0f, densePoint.values()[0], 0.001f);

        // Test single element sparse vector
        int[] sparseIndices = {0};
        float[] sparseValues = {42.0f};

        LabeledPoint sparsePoint = new LabeledPoint(label, size, sparseIndices, sparseValues);
        assertEquals(1, sparsePoint.indices().length);
        assertEquals(1, sparsePoint.values().length);
        assertEquals(0, sparsePoint.indices()[0]);
        assertEquals(42.0f, sparsePoint.values()[0], 0.001f);
    }
}
