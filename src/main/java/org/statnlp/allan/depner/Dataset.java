package org.statnlp.allan.depner;
/*
* 	@Author:  Danqi Chen
* 	@Email:  danqi@cs.stanford.edu
*	@Created:  2014-09-01
* 	@Last Modified:  2014-09-30
*/


import java.util.ArrayList;
import java.util.List;

/**
 * Defines a list of training / testing examples in multi-class classification
 * setting.
 *
 * @author Danqi Chen
 */

public class Dataset {

	public int n;
	public final int numFeatures, numLabels;
	public final List<Example> examples;

	public Dataset(int numFeatures, int numLabels) {
		n = 0;
		this.numFeatures = numFeatures;
		this.numLabels = numLabels;
		examples = new ArrayList<>();
	}

	public void addExample(List<Integer> feature, List<Integer> label) {
		Example data = new Example(feature, label);
		n += 1;
		examples.add(data);
	}

}
