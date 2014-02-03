package iais.network;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.TwoDimensionalMap;

/**
 * Outer context of a phrase approach: where left and right words adjacent to a phrase
 * are replaced by random words.
 * 
 * 
 * 
 * @author bhanu
 *
 */

public class RankingCostAndGradient2 extends AbstractCachingDiffFunction{

	RNNPhraseModel model;
	List<Tree> trainingBatch;

	public RankingCostAndGradient2(RNNPhraseModel model, List<Tree> trainingBatch){
		this.model = model;
		this.trainingBatch = trainingBatch;
	}

	@Override
	public int domainDimension() {
		return model.totalParamSize();
	}

	@Override
	protected void calculate(double[] theta) {
		model.vectorToParams(theta);

		TwoDimensionalMap<String, String, SimpleMatrix> encodeTD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.encodeTransform) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			encodeTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		TwoDimensionalMap<String, String, SimpleMatrix> decodeTD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.decodeTransform) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			decodeTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		TwoDimensionalMap<String, String, SimpleMatrix> wScoreD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.wScore) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			wScoreD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}
		
		TwoDimensionalMap<String, String, SimpleMatrix> wHiddenD = TwoDimensionalMap.treeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.wHidden) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			wHiddenD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		List<Tree> forwardPropTrees = Generics.newArrayList();

		if(model.gradientCheck)
			model.randContext = new Random(model.SEED); // reset random state, so that, random replacements are identical in each epoch

		for (int i=0; i<trainingBatch.size(); i++){	
			if(trainingBatch.get(i).isLeaf())
				continue;

			Tree tree = trainingBatch.get(i); //sentence tree
			//			List<Tree> phraseTrees = getPhraseTrees(tree, 2, 6);
			//			for(Tree pTree : phraseTrees){				

			Tree trainingTree = tree.deepCopy();
			//			List<Tree> leaves = trainingTree.getLeaves();
			try{
				forwardPropagateTree(trainingTree); //calculate nodevectors
			}catch(AssertionError e){
				//					System.out.println("Corrupt Tree, Caught Assertion Error :"+ e.toString());
				continue;
			}

			//add trees and nodes for backpropagation
			forwardPropTrees.add(trainingTree);

		}

		double error = 0.0;
		for (int i=0; i<forwardPropTrees.size();i++) {

			//accumulate derivates and error for ranking cost
			backpropRanking(forwardPropTrees.get(i), forwardPropTrees.get(i).getLeaves(), encodeTD, wScoreD, wHiddenD);

			//sum error accross all nodes
			double thisTreeError = sumError(forwardPropTrees.get(i));
			//			n_nodes = forwardPropTrees.get(i).getLeaves().size() - 1; 
			//			int nnodes = forwardPropTrees.get(i).size();
			//			thisTreeError /= n_nodes;
			error += thisTreeError;
		}

		// scale the error by the number of sentences so that the
		// regularization isn't drowned out for large training batchs
		//		double scale = (1.0 / trainingBatch.size());
		Integer nTrees = 1;
		if(forwardPropTrees.size() != 0)
			nTrees = forwardPropTrees.size();
		double scale = (1.0 / nTrees);
		value = error * scale;

		value += scaleAndRegularize(encodeTD, model.encodeTransform, scale, model.op.trainOptions.regTransform);
//		value += scaleAndRegularize(decodeTD, model.decodeTransform, scale, model.op.trainOptions.regTransform);
		value += scaleAndRegularize(wScoreD, model.wScore, scale, model.op.trainOptions.regClassification);
		value += scaleAndRegularize(wHiddenD, model.wHidden, scale, model.op.trainOptions.regTransform);

		derivative = RNNUtils.paramsToVector(theta.length, 
				encodeTD.valueIterator(),
				decodeTD.valueIterator(),
				wScoreD.valueIterator(),
				wHiddenD.valueIterator());

	}


	private void backpropRanking(Tree tree, List<Tree> leaves,
			TwoDimensionalMap<String, String, SimpleMatrix> encodeTD,
			TwoDimensionalMap<String, String, SimpleMatrix> wScoreD,
			TwoDimensionalMap<String, String, SimpleMatrix> wHiddenD){


		if(tree.isLeaf()){
			RNNCoreAnnotations.setRankingError(tree, 0.0);
			return;
		}
		else{

			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());
			SimpleMatrix wScore = model.getWScore(leftCategory, rightCategory);
			SimpleMatrix wHidden = model.getWHidden(leftCategory, rightCategory);

			SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);
			
			//calculate pos_score			
			SimpleMatrix leftContextVecsPos = getContextVec(tree, leaves, true, model.op.nWordsInContext);
			SimpleMatrix rightContextVecsPos = getContextVec(tree, leaves, false, model.op.nWordsInContext);
			SimpleMatrix posInput = RNNUtils.concatenate(leftContextVecsPos, nodeVector, rightContextVecsPos);	
			SimpleMatrix posHidden = wHidden.mult(RNNUtils.concatenateWithBias(posInput));
			posHidden = RNNUtils.elementwiseApplyTanh(posHidden);
			SimpleMatrix posScore = wScore.mult(RNNUtils.concatenateWithBias(posHidden));

			//calculate neg_score
			SimpleMatrix leftContextVecsNeg = getRandomContextVec(model.op.nWordsInContext, model.randContext);
			SimpleMatrix rightContextVecsNeg = getRandomContextVec(model.op.nWordsInContext, model.randContext);
			SimpleMatrix negInput = RNNUtils.concatenate(leftContextVecsNeg, nodeVector, rightContextVecsNeg);		
			SimpleMatrix negHidden = wHidden.mult(RNNUtils.concatenateWithBias(negInput));
			negHidden = RNNUtils.elementwiseApplyTanh(negHidden);
			SimpleMatrix negScore = wScore.mult(RNNUtils.concatenateWithBias(negHidden));

			//calcualte error
			// error = max(0, neg_score - pos_score + 1)
			double error = 0.0;
			if(1 - posScore.get(0,0) + negScore.get(0,0) > 0) 
				error = (1 - posScore.get(0,0) + negScore.get(0,0))  ; 
			else
				error = 0.0;		
			RNNCoreAnnotations.setRankingError(tree, error);

			//calculate wScoreD
			SimpleMatrix deltaMPos = new SimpleMatrix(wScore.numRows(), 1);
			SimpleMatrix deltaMNeg = new SimpleMatrix(wScore.numRows(), 1);
			if(error > 0){
				// deltaW_hm = delta_m*out_h; delta = -1 for pos example and delta = 1 for negative example				
				for(int i =0; i< deltaMPos.numRows(); i++){
					for(int j=0; j<deltaMPos.numCols(); j++) {
						deltaMPos.set(i,j, -1);
						deltaMNeg.set(i,j, 1);
					}
				}
			}
			//calculate wScore derivatives
			SimpleMatrix localPosD = deltaMPos.mult(RNNUtils.concatenateWithBias(posHidden).transpose());
			SimpleMatrix localNegD = deltaMNeg.mult(RNNUtils.concatenateWithBias(negHidden).transpose());
			SimpleMatrix localD = localPosD.plus(localNegD);
			wScoreD.put(leftCategory, rightCategory, wScoreD.get(leftCategory, rightCategory).plus(localD));
			
			//calculate wHidden derivatives
			SimpleMatrix posNodeVecDerivative = RNNUtils.elementwiseApplyTanhDerivative(posHidden);
			SimpleMatrix negNodeVecDerivative = RNNUtils.elementwiseApplyTanhDerivative(negHidden);
			SimpleMatrix deltaHPos = wScore.transpose().mult(deltaMPos);
			deltaHPos = deltaHPos.extractMatrix(0, model.nHidden, 0, 1).elementMult(posNodeVecDerivative);
			SimpleMatrix deltaHNeg = wScore.transpose().mult(deltaMNeg);
			deltaHNeg = deltaHNeg.extractMatrix(0, model.nHidden, 0, 1).elementMult(negNodeVecDerivative);
			SimpleMatrix hiddenPosD = deltaHPos.mult(RNNUtils.concatenateWithBias(posInput).transpose());
			SimpleMatrix hiddenNegD = deltaHNeg.mult(RNNUtils.concatenateWithBias(negInput).transpose());
			SimpleMatrix hiddenD = hiddenPosD.plus(hiddenNegD);
			wHiddenD.put(leftCategory, rightCategory, wHiddenD.get(leftCategory, rightCategory).plus(hiddenD));

			//calculate topDelta to be backpropagated through structure
			SimpleMatrix nodeVecDerivative = RNNUtils.elementwiseApplyTanhDerivative(nodeVector);
			SimpleMatrix deltaFromPosEx = wHidden.transpose().mult(deltaHPos);
			deltaFromPosEx = deltaFromPosEx.extractMatrix((model.op.nWordsInContext)*model.numHid, 
											(model.op.nWordsInContext+1)*model.op.numHid, 0, 1).elementMult(nodeVecDerivative);
			SimpleMatrix deltaFromNegEx = wHidden.transpose().mult(deltaHNeg);
			deltaFromNegEx = deltaFromNegEx.extractMatrix((model.op.nWordsInContext)*model.numHid, 
					(model.op.nWordsInContext+1)*model.op.numHid, 0, 1).elementMult(nodeVecDerivative);
			SimpleMatrix deltaUp = deltaFromPosEx.plus(deltaFromNegEx);
//			deltaUp = deltaUp.extractMatrix(model.op.nWordsInContext*model.numHid, (model.op.nWordsInContext+1)*model.numHid, 0, 1);

			//backprop this subtree for this phrase
			backpropSubTree(tree, leaves, encodeTD, deltaUp);

			//recursively switch to other nodes
			backpropRanking(tree.children()[0], leaves, encodeTD, wScoreD, wHiddenD);
			backpropRanking(tree.children()[1], leaves, encodeTD, wScoreD, wHiddenD);

		}

	}

	private void backpropSubTree(Tree tree, List<Tree> leaves,
			TwoDimensionalMap<String, String, SimpleMatrix> encodeTD,
			SimpleMatrix deltaUp
			) {		

		if(tree.isLeaf()){
			return;
		}
		else{

			//calculate deltah			
			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());

			//calculate W_df
			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector);
			SimpleMatrix W_df = deltaUp.mult(childrenVector.transpose());		
			encodeTD.put(leftCategory, rightCategory, encodeTD.get(leftCategory, rightCategory).plus(W_df));

			//calculate children deltas
			SimpleMatrix deltaDown;			
			deltaDown = model.getEncodeTransform(leftCategory, rightCategory).transpose().mult(deltaUp);
			SimpleMatrix leftDerivative = RNNUtils.elementwiseApplyTanhDerivative(leftVector);
			SimpleMatrix rightDerivative = RNNUtils.elementwiseApplyTanhDerivative(rightVector);
			SimpleMatrix leftDeltaDown = deltaDown.extractMatrix(0, deltaUp.numRows(), 0, 1);
			SimpleMatrix rightDeltaDown = deltaDown.extractMatrix(deltaUp.numRows(), deltaUp.numRows() * 2, 0, 1);

			//backprop recursively to children
			backpropSubTree(tree.children()[0],  leaves, encodeTD, leftDerivative.elementMult(leftDeltaDown));
			backpropSubTree(tree.children()[1],  leaves, encodeTD, rightDerivative.elementMult(rightDeltaDown));

		}

	}


	public double sumError(Tree tree) {
		if (tree.isLeaf()) {
			return RNNCoreAnnotations.getRankingError(tree); //rec error for leaves is 0.0

		} else {
			double error = 0.0;
			for (Tree child : tree.children()) {
				error += sumError(child);
			}
			return RNNCoreAnnotations.getRankingError(tree) + error;
		}
	}

	public double scaleAndRegularize(TwoDimensionalMap<String, String, SimpleMatrix> derivatives,
			TwoDimensionalMap<String, String, SimpleMatrix> currentMatrices,
			double scale,
			double regCost) {
		double cost = 0.0; // the regularization cost
		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : currentMatrices) {
			SimpleMatrix D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
			D = D.scale(scale).plus(entry.getValue().scale(regCost));
			derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
			cost += entry.getValue().elementMult(entry.getValue()).elementSum() * regCost / 2.0;
		}
		return cost;
	}

	public List<Tree> getPhraseTrees(Tree tree, Integer minPhLen, Integer maxPhLen) {
		List<Tree> phraseTrees = Generics.newArrayList();
		Iterator<Tree> nodes = tree.iterator();
		while(nodes.hasNext()){
			Tree node = nodes.next();
			if(node.getLeaves().size() >= minPhLen  && node.getLeaves().size() <= maxPhLen){
				phraseTrees.add(node);
			}		
		}
		return phraseTrees;
	}


	/**
	 * This is the method to call for assigning  node vectors
	 * to the Tree.  After calling this, each of the non-leaf nodes will
	 * have the node vector.  The annotations filled in are
	 * the RNNCoreAnnotations.NodeVector. 
	 * @param leaves 
	 */
	public void forwardPropagateTree(Tree tree) {
		SimpleMatrix nodeVector = null;


		if (tree.isLeaf()) {			
			String word = tree.label().value();	
			if(word == null)
				throw new AssertionError("Tree containing a null word");
			nodeVector = model.getWordVector(word);	

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			forwardPropagateTree(tree.children()[0]);
			forwardPropagateTree(tree.children()[1]);

			String leftCategory = tree.children()[0].label().value();
			String rightCategory = tree.children()[1].label().value();
			SimpleMatrix W = model.getEncodeTransform(leftCategory, rightCategory);			

			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = null;


			childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector );


			nodeVector = RNNUtils.elementwiseApplyTanh(W.mult(childrenVector));			

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}

		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);

	}


	private SimpleMatrix getRandomContextVec(int nWordsInContext, Random rand) {
		if (nWordsInContext < 1) {
			throw new AssertionError("Number of words in outer context should be atleast 1");
		}
		
		SimpleMatrix contextVecs = null;
		String word ;
		if(rand==null){
			word = model.getRandomVocabWord(model.rand);
		}else
			word = model.getRandomVocabWord(rand);
		contextVecs = model.getWordVector(word);	
		for(int i=1; i<nWordsInContext; i++){

			if(rand==null){
				word = model.getRandomVocabWord(model.rand);
			}else
				word = model.getRandomVocabWord(rand);

			contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
		}
//		contextVecs = RNNUtils.elementwiseApplyTanh(contextVecs);
		return contextVecs;
	}

	private SimpleMatrix getContextVec(Tree tree, List<Tree> leaves, boolean isLeftContext, int nWordsInContext) {
		if (nWordsInContext < 1) {
			throw new AssertionError("Number of words in outer context should be atleast 1");
		}
		
		SimpleMatrix contextVecs = null;
		List<Tree> spannedLeaves = tree.getLeaves();
		Tree leafToUse = null;
		CoreLabel label = null;
		String word = null;
		if(isLeftContext){
			leafToUse = spannedLeaves.get(0);
			label = (CoreLabel)leafToUse.label();
			Integer nodeId = label.get(IndexAnnotation.class) - 1; // node indexes starts from 1
			word = "";
			if(nodeId - 1 < 0)
				word = RNNPhraseModel.PADDING;
			else
				word = leaves.get(nodeId-1).label().value();
			contextVecs = model.getWordVector(word);	
			for(int i=1; i<nWordsInContext; i++){				
				if(nodeId - (i+1) < 0)
					word = RNNPhraseModel.PADDING;
				else
					word = leaves.get(nodeId-(i+1)).label().value();
				contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
			}
		}else{
			leafToUse = spannedLeaves.get(spannedLeaves.size()-1);
			label = (CoreLabel)leafToUse.label();
			Integer nodeId = label.get(IndexAnnotation.class) - 1; // node indexes starts from 1
			word = "";
			if(nodeId +1 > leaves.size()-1)
				word = RNNPhraseModel.PADDING;
			else
				word = leaves.get(nodeId+1).label().value();
			contextVecs = model.getWordVector(word);	
			for(int i=1; i<nWordsInContext; i++){				
				if(nodeId + (i+1) >leaves.size()-1)
					word = RNNPhraseModel.PADDING;
				else
					word = leaves.get(nodeId+(i+1)).label().value();
				contextVecs = RNNUtils.concatenate(contextVecs, model.getWordVector(word));
			}
		}

//		contextVecs = RNNUtils.elementwiseApplyTanh(contextVecs);
		return contextVecs;
	}




}
