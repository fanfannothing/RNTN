package iais.network;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.TwoDimensionalMap;

public class ReconRankingCostAndGradient extends AbstractCachingDiffFunction{

	RNNPhraseModel model;
	List<Tree> trainingBatch;

	public ReconRankingCostAndGradient(RNNPhraseModel model, List<Tree> trainingBatch){
		this.model = model;
		this.trainingBatch = trainingBatch;
	}

	@Override
	public int domainDimension() {
		return model.totalParamSize();
	}

	@Override
	protected void calculate(double[] theta) {
		ReconstructionCostAndGradient reConFunc = new ReconstructionCostAndGradient(model, trainingBatch);
//		RankingCostAndGradient rankingFun2 = new RankingCostAndGradient(model, trainingBatch);
		RankingCostAndGradient2 rankingFun2 = new RankingCostAndGradient2(model, trainingBatch); //using context based 
	
		double reconCost = 0;
		double rankingCost = 0;

		// TODO: do we want to iterate multiple times per batch?
		double[] reconGrad = reConFunc.derivativeAt(theta);
		reconCost = reConFunc.valueAt(theta);
		
		double[] rankingGrad = rankingFun2.derivativeAt(theta);
		rankingCost = rankingFun2.valueAt(theta);
		
		double[] gradf = new double[reconGrad.length];
		for(int i=0; i< reconGrad.length; i++){
			gradf[i] = model.alpha*rankingGrad[i] + (1-model.alpha)*reconGrad[i];
		}
		value = model.alpha*rankingCost + (1-model.alpha)*reconCost;
		derivative = gradf;

	}


}
