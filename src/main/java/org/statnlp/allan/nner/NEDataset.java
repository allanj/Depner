package org.statnlp.allan.nner;
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

public class NEDataset {

	public int n;
	public final int numFeatures, numLabels;
	public final List<NEExample> examples;

	public NEDataset(int numFeatures, int numLabels) {
		n = 0;
		this.numFeatures = numFeatures;
		this.numLabels = numLabels;
		examples = new ArrayList<>();
	}

	public void addExample(List<List<Integer>> features, List<String> actionSeq) {
		NEExample data = new NEExample(features, actionSeq);
		n += 1;
		examples.add(data);
	}

}
