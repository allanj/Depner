package org.statnlp.allan.nner;

import java.util.HashMap;
import java.util.Random;

import org.statnlp.allan.depner.Dataset;

import com.ml.common.Instance;
import com.ml.process.Processer;
import com.ml.utils.Config;
import com.ml.utils.LossFunction;

public class StructuredPerceptron {

	/**
	 * Weight vector of the perceptron
	 */
	private double[] weight;
	
	private int maxIter;
	
	private boolean isAvg ;
	
	private double[] avgWeight;
	
	private int numOutputs;
	
	
	private HashMap<String, Integer> str2int;
	
	private final Dataset trainingData;
	
	/**
	 * Constructor of Perceptron
	 * @param dimension: dimension of x
	 * @param maxIteration: the maxNumber of Iteration
	 * @param LossFunction: zero_loss or hinge loss in the LossFunction class
	 * @param learningRate: the learningRate for SGD if you use the hinge loss which apply SGD to train
	 */
	public StructuredPerceptron(int featureNum, int maxIteration,int numOutputs, Dataset trainInsts) {
		weight = new double[featureNum];
		maxIter = maxIteration;
		this.numOutputs = numOutputs;
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
	}
	
	
	public void train(){
		for (int iter = 1; iter <= this.maxIter; iter++) {
			for (int idx = 0; idx < this.trainingData.n; idx++) {
				
			}
		}
	}
	
	public void trainZeroOne(Instance[] insts){
		int iterCount = 0;
		while(true){
			int error = 0;
			for(Instance inst: insts){
				double[] x = inst.getInput();
				double sum =0;
				for(int i=0;i<x.length;i++)
					sum+=weight[i+1]*x[i];
				sum+=weight[0]*1;
				int pred = Math.signum(sum)>=0? 1:-1;
				if(str2int.get(inst.getStrLabel()).intValue()!=pred){
					weight[0] = weight[0] + str2int.get(inst.getStrLabel()).intValue();
					for(int k=1;k<weight.length;k++){
						weight[k] = weight[k] + str2int.get(inst.getStrLabel()).intValue()*x[k-1];
					}
					error++;
				}
				if(this.isAvg){
					for(int k=0;k<weight.length;k++)
						avgWeight[k] += weight[k];
				}
			}
			if(Config.DEBUG)
				System.err.println("[DEBUG] Training Error now:"+error);
			iterCount++;
			if(iterCount==maxIter || error==0){
				if(this.isAvg) {
					for(int k=0;k<weight.length;k++)
						weight[k] = avgWeight[k]/(iterCount*insts.length);
				}
				break;
			}
		}
	}
	
	public void trainWithSGD(Instance[] insts, double learningRate){
		System.err.println("[Info] Training with Stochastic Gradient Descent");
		System.err.println("[Info] Iteration:"+this.maxIter);
		int iterCount = 0;
		Random rand = new Random();
		double[] bestWeight = new double[weight.length]; 
		double lowestRisk = Double.MAX_VALUE;
		while(true){
			Instance inst = insts[rand.nextInt(insts.length)];
			double[] x = inst.getInput();
			double sum =0;
			for(int i=0;i<x.length;i++)
				sum+=weight[i+1]*x[i];
			sum+=weight[0]*1;
			if(sum*str2int.get(inst.getStrLabel())<=1){
				weight[0] = weight[0] + learningRate*str2int.get(inst.getStrLabel());
				for(int k=1;k<weight.length;k++){
					weight[k] = weight[k] + learningRate*str2int.get(inst.getStrLabel())*x[k-1]-2*reg*learningRate*weight[k];
				}
			}else{
				for(int k=1;k<weight.length;k++){
					weight[k] = weight[k] -2*reg*learningRate*weight[k];
				}
			}
			double risk = 0;
			for(int i=0;i<insts.length;i++){
				double sumLoss = 0;
				double[] xx = insts[i].getInput();
				for(int k=0;k<xx.length;k++)
					sumLoss+=weight[k+1]*xx[k];
				sumLoss+=weight[0]*1;
				risk+=Math.max(0, 1-str2int.get(insts[i].getStrLabel())*sumLoss);
			}
			risk/=insts.length;
			double sumW = 0;
			for(int k=1;k<weight.length;k++){
				sumW+=weight[k]*weight[k];
			}
			risk+=sumW*reg;
			if(risk<=lowestRisk){
				for(int kk=0;kk<weight.length;kk++)
					bestWeight[kk] = weight[kk];
			}
			iterCount++;
			if(risk==0.0)
				System.err.println("[Info] Current risk is zero, converge. Iteration Number:"+iterCount);
			if(iterCount==maxIter || risk==0.0){
				break;
			}
		}
		weight = bestWeight;
	}

	
	public void trainWithModifiedSGD(Instance[] insts, double learningRate){
		int iterCount = 0;
		Random rand = new Random();
		double[] bestWeight = new double[weight.length]; 
		double lowestRisk = Double.MAX_VALUE;
		while(true){
			Instance inst = insts[rand.nextInt(insts.length)];
			double[] x = inst.getInput();
			double sum =0;
			for(int i=0;i<x.length;i++)
				sum+=weight[i+1]*x[i];
			sum+=weight[0]*1;
			if(sum*str2int.get(inst.getStrLabel())<=1){
				double sumDot = sum;
//				System.err.println(sumDot);
				sumDot-=str2int.get(inst.getStrLabel());
				for(int k=1;k<weight.length;k++){
					weight[k] = weight[k] - 2*learningRate*sumDot*x[k-1];
				}
				weight[0] = weight[0] - 2*learningRate*(weight[0]-str2int.get(inst.getStrLabel()));
//				System.err.println(Arrays.toString(weight));
			}
			double risk = 0;
			for(int i=0;i<insts.length;i++){
				double sumLoss = 0;
				double[] xx = insts[i].getInput();
				for(int k=0;k<xx.length;k++)
					sumLoss+=weight[k+1]*xx[k];
				sumLoss+=weight[0]*1;
				double oneLoss = sumLoss*str2int.get(insts[i].getStrLabel());
				if(oneLoss<=1){
					risk+=(1-oneLoss)*(1-oneLoss);
				}
			}
			risk/=insts.length;
			if(risk<=lowestRisk){
				for(int kk=0;kk<weight.length;kk++)
					bestWeight[kk] = weight[kk];
			}
			iterCount++;
			if(risk==0.0)
				System.err.println("[Info] Current risk is zero, converge. Iteration Number:"+iterCount);
			if(iterCount==maxIter || risk==0.0){
				break;
			}
		}
		weight = bestWeight;
	}

	
	
	public double test(Instance[][] testInsts){
		int errors = 0;
		int total = 0;
		for(int t=0;t<testInsts.length;t++){
			for(Instance inst: testInsts[t]){
				double[] x = inst.getInput();
				double sum =0;
				for(int i=0;i<x.length;i++)
					sum+=weight[i+1]*x[i];
				sum+=weight[0]*1;
				int pred = Math.signum(sum)>=0? 1:-1;
				String predStr = pred==1? types[0]:types[1];
				inst.setPredStrLabel(predStr);
				if(!inst.getStrLabel().equals(predStr))
					errors++;
				total++;
			}
		}
		System.err.println("[Info] Perceptron: Total Number of Testing Errors:"+errors);
		System.err.println("[Info] Perceptron Accuracy:"+(total-errors)*1.0/total);
		return ((total-errors)*1.0/total);
	}
}
