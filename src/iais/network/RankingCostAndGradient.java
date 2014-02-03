package iais.network;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.rnn.SimpleTensor;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.TwoDimensionalMap;

/**
 * Inner context approach: where an illegal phrase is produced by replacing the middle most leaf node
 * with a random word.
 * 
 * @author bhanu
 *
 */
public class RankingCostAndGradient extends AbstractCachingDiffFunction{

	RNNPhraseModel model;
	List<Tree> trainingBatch;

	public RankingCostAndGradient(RNNPhraseModel model, List<Tree> trainingBatch){
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

		List<Tree> forwardPropTrees = Generics.newArrayList();

		if(model.gradientCheck)
			model.randContext = new Random(42);
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

		//backpropagate error and derivatives for each word, verb example
		//currently assuming that batch size should be one, i.e. one sentence 
		int n_nodes = 0;
		double error = 0.0;
		for (int i=0; i<forwardPropTrees.size();i++) {

			//accumulate derivates and error for ranking cost
			backpropRanking(forwardPropTrees.get(i), forwardPropTrees.get(i).getLeaves(), encodeTD, wScoreD);

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
		value += scaleAndRegularize(decodeTD, model.decodeTransform, scale, model.op.trainOptions.regTransform);
		value += scaleAndRegularize(wScoreD, model.wScore, scale, model.op.trainOptions.regClassification);

		derivative = RNNUtils.paramsToVector(theta.length, 
				encodeTD.valueIterator(),
				decodeTD.valueIterator(),
				wScoreD.valueIterator());

	}

	private void backpropRanking(Tree tree, List<Tree> leaves,
			TwoDimensionalMap<String, String, SimpleMatrix> encodeTD,
			TwoDimensionalMap<String, String, SimpleMatrix> wScoreD){

		//		forwardPropagateTreeNeg(tree); //attaches  nodevectors for random context to each non-terminal node
		SimpleMatrix deltaPos = new SimpleMatrix(model.op.numHid, 1);
		SimpleMatrix deltaNeg = new SimpleMatrix(model.op.numHid, 1);
		Tree corruptTree = getCorruptTree(tree, model.randContext);
		forwardPropagateTree(corruptTree);
		backpropRanking(tree, corruptTree, leaves, encodeTD, wScoreD, deltaPos, deltaNeg);

	}

	private void backpropRanking(Tree tree, Tree corruptTree, List<Tree> leaves,
			TwoDimensionalMap<String, String, SimpleMatrix> encodeTD,
			TwoDimensionalMap<String, String, SimpleMatrix> wScoreD,
			SimpleMatrix deltaUpPos,
			SimpleMatrix deltaUpNeg) {		

		SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);
		//		SimpleMatrix nodeVectorNeg = RNNCoreAnnotations.getNodeVectorNeg(tree);
		double error = 0.0;

		if(tree.isLeaf()){
			RNNCoreAnnotations.setRankingError(tree, error);
			return;
		}
		else{
			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());
			SimpleMatrix WScore = model.getWScore(leftCategory, rightCategory);

//			SimpleMatrix leftContextVecs = getContextVec(tree, leaves, true, model.op.nWordsInContext);
//			SimpleMatrix rightContextVecs = getContextVec(tree, leaves, false, model.op.nWordsInContext);
//			SimpleMatrix posInput = RNNUtils.concatenate(leftContextVecs, nodeVector, rightContextVecs);
			SimpleMatrix posInput = nodeVector;
			SimpleMatrix pos_scoreMat = WScore.mult(RNNUtils.concatenateWithBias(nodeVector));
//			pos_scoreMat = RNNUtils.elementwiseApplyTanh(pos_scoreMat);
			double pos_score = pos_scoreMat.get(0, 0);


//			SimpleMatrix randLeftContextVecs = getRandomContextVec(model.op.nWordsInContext);
//			SimpleMatrix randRightContextVecs = getRandomContextVec(model.op.nWordsInContext);
//			SimpleMatrix negInput = RNNUtils.concatenate(randLeftContextVecs, nodeVector, randRightContextVecs);
//			SimpleMatrix negInput = getCorruptNodeVector(tree, model.randContext);
//			Tree corruptTree = getCorruptTree(tree, model.randContext);
//			forwardPropagateTree(corruptTree);
			SimpleMatrix negInput = RNNCoreAnnotations.getNodeVector(corruptTree);
			SimpleMatrix neg_scoreMat = WScore.mult(RNNUtils.concatenateWithBias(negInput));
//			neg_scoreMat = RNNUtils.elementwiseApplyTanh(neg_scoreMat);
			double neg_score = neg_scoreMat.get(0, 0);

			// error = max(0, neg_score - pos_score + 1)
			if(1 - pos_score + neg_score > 0) 
				error = (1 - pos_score + neg_score)  ; 
			else
				error = 0.0;		
			RNNCoreAnnotations.setRankingError(tree, error);

			SimpleMatrix deltaMPos = new SimpleMatrix(WScore.numRows(), 1);
			SimpleMatrix deltaMNeg = new SimpleMatrix(WScore.numRows(), 1);
			if(error > 0){
//				SimpleMatrix posScoreDerivative = RNNUtils.elementwiseApplyTanhDerivative(pos_scoreMat);
//				SimpleMatrix negScoreDerivative = RNNUtils.elementwiseApplyTanhDerivative(neg_scoreMat);
				//calculate wScoreD
				// deltaW_hm = delta_m*out_h; delta= -1 for pos example and 1 for negative example				
				for(int i =0; i< deltaMPos.numRows(); i++){
					for(int j=0; j<deltaMPos.numCols(); j++) {//no change for bias
						deltaMPos.set(i,j, -1);
						deltaMNeg.set(i,j, 1);
					}
				}
//				deltaMPos = deltaMPos.elementMult(posScoreDerivative);
//				deltaMNeg = deltaMNeg.elementMult(negScoreDerivative);
			}

			SimpleMatrix localPosD = deltaMPos.mult(RNNUtils.concatenateWithBias(posInput).transpose());
			SimpleMatrix localNegD = deltaMNeg.mult(RNNUtils.concatenateWithBias(negInput).transpose());
			SimpleMatrix localD = localPosD.plus(localNegD);
//			localD = localD.scale(0.5);
			wScoreD.put(leftCategory, rightCategory, wScoreD.get(leftCategory, rightCategory).plus(localD));

			//calculate deltah				
			SimpleMatrix posNodeVecDerivative = RNNUtils.elementwiseApplyTanhDerivative(posInput);
			SimpleMatrix negNodeVecDerivative = RNNUtils.elementwiseApplyTanhDerivative(negInput);
			SimpleMatrix deltaFromPosEx = WScore.transpose().mult(deltaMPos);
			deltaFromPosEx = deltaFromPosEx.extractMatrix(0, model.op.numHid, 0, 1).elementMult(posNodeVecDerivative);
			SimpleMatrix deltaFromNegEx = WScore.transpose().mult(deltaMNeg);
			deltaFromNegEx = deltaFromNegEx.extractMatrix(0, model.op.numHid, 0, 1).elementMult(negNodeVecDerivative);
			SimpleMatrix deltaFullPos = deltaFromPosEx.plus(deltaUpPos);	
			SimpleMatrix deltaFullNeg = deltaFromNegEx.plus(deltaUpNeg);
//			deltaFull = deltaFull.extractMatrix(model.op.nWordsInContext*model.numHid, (model.op.nWordsInContext+1)*model.numHid, 0, 1).plus(deltaUp);
//			deltaFull = deltaFull.plus(deltaUp);

			//calculate W_df
			SimpleMatrix leftVectorPos = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVectorPos = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVectorPos = RNNUtils.concatenateWithBias(leftVectorPos, rightVectorPos);
			SimpleMatrix W_df_Pos = deltaFullPos.mult(childrenVectorPos.transpose());
			
			SimpleMatrix leftVectorNeg = RNNCoreAnnotations.getNodeVector(corruptTree.children()[0]);
			SimpleMatrix rightVectorNeg = RNNCoreAnnotations.getNodeVector(corruptTree.children()[1]);
			SimpleMatrix childrenVectorNeg = RNNUtils.concatenateWithBias(leftVectorNeg, rightVectorNeg);
			SimpleMatrix W_df_Neg = deltaFullNeg.mult(childrenVectorNeg.transpose());
			
			SimpleMatrix W_df = W_df_Pos.plus(W_df_Neg);
//			W_df = W_df.scale(0.5);
			encodeTD.put(leftCategory, rightCategory, encodeTD.get(leftCategory, rightCategory).plus(W_df));
			
			

			//calculate children deltas
			SimpleMatrix deltaDownPos;			
			deltaDownPos = model.getEncodeTransform(leftCategory, rightCategory).transpose().mult(deltaFullPos);
			SimpleMatrix leftDerivativePos = RNNUtils.elementwiseApplyTanhDerivative(leftVectorPos);
			SimpleMatrix rightDerivativePos = RNNUtils.elementwiseApplyTanhDerivative(rightVectorPos);
			SimpleMatrix leftDeltaDownPos = deltaDownPos.extractMatrix(0, deltaFullPos.numRows(), 0, 1);
			SimpleMatrix rightDeltaDownPos = deltaDownPos.extractMatrix(deltaFullPos.numRows(), deltaFullPos.numRows() * 2, 0, 1);
			
			SimpleMatrix deltaDownNeg;			
			deltaDownNeg = model.getEncodeTransform(leftCategory, rightCategory).transpose().mult(deltaFullNeg);
			SimpleMatrix leftDerivativeNeg = RNNUtils.elementwiseApplyTanhDerivative(leftVectorNeg);
			SimpleMatrix rightDerivativeNeg = RNNUtils.elementwiseApplyTanhDerivative(rightVectorNeg);
			SimpleMatrix leftDeltaDownNeg = deltaDownNeg.extractMatrix(0, deltaFullNeg.numRows(), 0, 1);
			SimpleMatrix rightDeltaDownNeg = deltaDownNeg.extractMatrix(deltaFullNeg.numRows(), deltaFullNeg.numRows() * 2, 0, 1);
			
			//backprop recursively to children
			backpropRanking(tree.children()[0], corruptTree.children()[0], leaves, encodeTD, wScoreD, leftDerivativePos.elementMult(leftDeltaDownPos), 
					leftDerivativeNeg.elementMult(leftDeltaDownNeg));
			backpropRanking(tree.children()[1], corruptTree.children()[1], leaves, encodeTD, wScoreD, rightDerivativePos.elementMult(rightDeltaDownPos), 
					rightDerivativeNeg.elementMult(rightDeltaDownNeg));


		}

	}


	private Tree getCorruptTree(Tree tree, Random rand){
		Tree corruptTree = tree.deepCopy();
		List<Tree> leaves = corruptTree.getLeaves();
		int corruptLeafId = leaves.size()/2;
		String randWord = "UNKNOWN";
		if(rand==null)
			randWord = model.getRandomVocabWord();
		else
			randWord = model.getRandomVocabWord(rand);
		
		CoreLabel corruptLeafLabel = (CoreLabel)leaves.get(corruptLeafId).label();
		corruptLeafLabel.setValue(randWord);
		
//		forwardPropagateTree(corruptTree);
		
		return corruptTree;
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



}
