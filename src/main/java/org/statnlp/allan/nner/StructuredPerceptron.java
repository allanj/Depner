package org.statnlp.allan.nner;

import java.util.HashMap;
import java.util.List;


public class StructuredPerceptron {

	/**
	 * Weight vector of the perceptron
	 */
	private double[] weight;
	
	private int maxIter;
	
	private boolean isAvg ;
	
	private double[] avgWeight;
	private double[] allWeight;
	private int avgCount;
	
	
	public final NEDataset trainingData;
	
	private NEConfig config;
	
	/**
	 * Constructor of Perceptron
	 * @param dimension: dimension of x
	 * @param maxIteration: the maxNumber of Iteration
	 * @param LossFunction: zero_loss or hinge loss in the LossFunction class
	 * @param learningRate: the learningRate for SGD if you use the hinge loss which apply SGD to train
	 */
	public StructuredPerceptron(int featureNum, int maxIteration,int numOutputs, NEDataset trainInsts) {
		weight = new double[featureNum];
		maxIter = maxIteration;
		this.trainingData = trainInsts;
		System.err.println("[Info] Maximum Iteration:"+maxIter);
	}
	
	public StructuredPerceptron(NEConfig config, double[] weights,int numOutputs, NEDataset trainInsts) {
		this.config = config;
		this.weight = weights;
		this.trainingData = trainInsts;
		System.err.println("[Info] Maximum Iteration:"+maxIter);
	}
	
	
	/**
	 * Enable the average perceptron algorithm
	 */
	public void enableAvgPerceptron(){
		System.err.println("[Info] Enabling the average perceptron.");
		isAvg = true;
		avgWeight = new double[weight.length];
		allWeight = new double[weight.length];
		avgCount = 0;
	}
	
	public void update(List<Integer> goldFeatures, List<Integer> predFeatures) {
		for (int f: goldFeatures)
			if (f != -1)
			this.weight[f]++;
		for (int f: predFeatures)
			if (f != -1)
			this.weight[f]--;
	}
	
	public void incrementAvgWeight() {
		for (int w = 0; w < this.weight.length; w++)
			this.allWeight[w] += this.weight[w];
		avgCount++;
	}
	
	public void averageWeight() {
		for (int w = 0; w < this.weight.length; w++)
			this.avgWeight[w] = this.allWeight[w]/avgCount;
	}
	
	public double getScore (List<Integer> feature, boolean isTrain) {
		double sum = 0;
		for (int f: feature)
			if (f != -1) {
				if (!isTrain && isAvg) {
					sum += this.avgWeight[f];
				} else {
					sum += weight[f];
				}
			}
				
				
		return sum;
	}
	
	
	public double[] getWeights() {
		if (this.isAvg)
			return this.avgWeight;
		else return this.weight;
	}
	
	public void finalizeClassifier() {
		if (this.isAvg)
			this.weight = this.avgWeight;
		this.allWeight = null;
	}
}
