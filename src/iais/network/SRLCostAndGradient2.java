package iais.network;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.Categories;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.Labels;
import edu.stanford.nlp.rnn.RNNCoreAnnotations.VerbIds;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.rnn.SimpleTensor;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.TwoDimensionalMap;


/**
 * Implementation of "all-nodes tagging" approach where we classify each node(including non-terminal nodes)
 * First each node is assigned a label : O(outside) or semantic role(A0-A5, etc) 
 * Then, each node is classified using the features:
 * 
 * top-node(top node in the syntactic path between this node and verb-node)
 * verb-node(predicate node)
 * this-node(node being classified)
 * 
 * @author bhanu
 *
 */

public class SRLCostAndGradient2 extends AbstractCachingDiffFunction {
	RNTNModel model;
	List<Tree> trainingBatch;
	List<VerbIds> verbIndices;
	List<Labels> sentenceLabels;	

	static final Integer topNodeId = 0; //node indexes start from 1
	static final Integer paddingNodeId = -1;
	static final Integer phraseNodeId = -2;

	Categories iobCategories = Categories.getIOBCategories();
	Categories roleCats = Categories.getRoleCats(iobCategories);

	public SRLCostAndGradient2(RNTNModel model, List<Tree> trainingBatch) {
		this.model = model;
		this.trainingBatch = trainingBatch;

		if(trainingBatch != null){
			this.verbIndices = Generics.newArrayList();
			this.sentenceLabels = Generics.newArrayList();
			for (Tree tree : trainingBatch){
				CoreLabel label  = (CoreLabel)tree.label();				
				this.verbIndices.add(label.get(RNNCoreAnnotations.VerbIdsAnnotation.class));				
				this.sentenceLabels.add(label.get(RNNCoreAnnotations.TagsAnnotation.class));
			}
		}
	}

	public int domainDimension() {
		// TODO: cache this for speed?
		return model.totalParamSize();
	}

	public double sumError(Tree tree) {
		if (tree.isLeaf()) {
			return RNNCoreAnnotations.getPredictionError(tree);
		} else if (tree.isPreTerminal()) {
			return RNNCoreAnnotations.getPredictionError(tree);
		} else {
			double error = 0.0;
			for (Tree child : tree.children()) {
				error += sumError(child);
			}
			return RNNCoreAnnotations.getPredictionError(tree) + error;
		}
	}

	/**
	 * Returns the index with the highest value in the <code>predictions</code> matrix.
	 * Indexed from 0.
	 */
	public int getPredictedClass(SimpleMatrix predictions) {
		int argmax = 0;
		for (int i = 1; i < predictions.getNumElements(); ++i) {
			if (predictions.get(i) > predictions.get(argmax)) {
				argmax = i;
			}
		}
		return argmax;
	}

	public void calculate(double[] theta) {
		model.vectorToParams(theta);

		// We use TreeMap for each of these so that they stay in a
		// canonical sorted order
		// TODO: factor out the initialization routines
		// binaryTD stands for Transform Derivatives (see the SRLModel)
		TwoDimensionalMap<String, String, SimpleMatrix> binaryTD = TwoDimensionalMap.treeMap();
		// the derivatives of the tensors for the binary nodes
		TwoDimensionalMap<String, String, SimpleTensor> binaryTensorTD = TwoDimensionalMap.treeMap();
		// binaryCD stands for Classification Derivatives
		TwoDimensionalMap<String, String, SimpleMatrix> binaryCD = TwoDimensionalMap.treeMap();

		// hidden layer for classification, wHiddenD stands for derivatives hidden layer weights derivatives
		TwoDimensionalMap<String, String, SimpleMatrix> wHiddenD = TwoDimensionalMap.treeMap();

		// unaryCD stands for Classification Derivatives
		Map<String, SimpleMatrix> unaryCD = Generics.newTreeMap();

		// word vector derivatives
		Map<String, SimpleMatrix> wordVectorD = Generics.newTreeMap();

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.binaryTransform) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			binaryTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		if (!model.op.combineClassification) {
			for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.binaryClassification) {
				int numRows = entry.getValue().numRows();
				int numCols = entry.getValue().numCols();

				binaryCD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
			}
		}

		if (model.op.useTensors) {
			for (TwoDimensionalMap.Entry<String, String, SimpleTensor> entry : model.binaryTensors) {
				int numRows = entry.getValue().numRows();
				int numCols = entry.getValue().numCols();
				int numSlices = entry.getValue().numSlices();

				binaryTensorTD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleTensor(numRows, numCols, numSlices));
			}
		}

		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : model.wHidden) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();

			wHiddenD.put(entry.getFirstKey(), entry.getSecondKey(), new SimpleMatrix(numRows, numCols));
		}

		for (Map.Entry<String, SimpleMatrix> entry : model.unaryClassification.entrySet()) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();
			unaryCD.put(entry.getKey(), new SimpleMatrix(numRows, numCols));
		}
		for (Map.Entry<String, SimpleMatrix> entry : model.wordVectors.entrySet()) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();
			wordVectorD.put(entry.getKey(), new SimpleMatrix(numRows, numCols));
		}


		// TODO: This part can easily be parallelized
		List<Tree> forwardPropTrees = Generics.newArrayList();
		//		List<Tree> subtrees = Generics.newArrayList();
		//		List<List<Integer>> inputNodeIds = Generics.newArrayList();
		List<Tree> verbNodes = Generics.newArrayList();
		//		List<Tree> wordNodes  = Generics.newArrayList();
		//		List<Integer> distances = Generics.newArrayList();
		double error = 0.0;
		for (int i=0; i<trainingBatch.size(); i++){	
			if((verbIndices.get(i).size() == 0) || trainingBatch.get(i).isLeaf())
				continue;
			for(int nverbs=0; nverbs < verbIndices.get(i).size(); nverbs++){// verbIndex : verbIndices.get(i)){
				Tree tree = trainingBatch.get(i); //sentence tree
				Integer verbIndex = verbIndices.get(i).get(nverbs); //verbindex for this sentence
				List<Tree> leaves = tree.getLeaves();
				if(sentenceLabels.get(i).get(nverbs).size() != leaves.size())
					continue;  //sentence tokens and labels mismatch.. have to fix this


				Tree trainingTree = tree.deepCopy();
				setVerbNWordIndexFeatures(trainingTree, verbIndex);
				//				setAdditionalFeatures(trainingTree, verbIndex);
				List<Integer> trueLabels = sentenceLabels.get(i).get(nverbs);
				attachLabelsToNodes(trainingTree, trueLabels, iobCategories, roleCats);
				Tree verbNode = trainingTree.getLeaves().get(verbIndex);
				forwardPropagateTree(trainingTree, trainingTree, verbNode); //calculate nodevectors
				attachPredictions(trainingTree, trainingTree, verbNode);
				//				trainingTree.pennPrint();
				forwardPropTrees.add(trainingTree);
				verbNodes.add(verbNode);

			}
			
		}  

		//backpropagate error and derivatives for each word, verb example
		//currently assuming that batch size should be one, i.e. one sentence 
		for (Integer i=0; i< forwardPropTrees.size(); i++) {

			backpropDerivativesAndError(forwardPropTrees.get(i), verbNodes.get(i),
					binaryTD, binaryCD, binaryTensorTD, unaryCD, wHiddenD, wordVectorD);
			error += sumError(forwardPropTrees.get(i));
		}

		// scale the error by the number of sentences so that the
		// regularization isn't drowned out for large training batchs
		//		double scale = (1.0 / trainingBatch.size());
		Integer nTrees = 1;
		if(forwardPropTrees.size() != 0)
			nTrees = forwardPropTrees.size();
		double scale = (1.0 / nTrees);
		value = error * scale;

		value += scaleAndRegularize(binaryTD, model.binaryTransform, scale, model.op.trainOptions.regTransform);
		value += scaleAndRegularize(binaryCD, model.binaryClassification, scale, model.op.trainOptions.regClassification);
		value += scaleAndRegularizeTensor(binaryTensorTD, model.binaryTensors, scale, model.op.trainOptions.regTransform);
		value += scaleAndRegularize(unaryCD, model.unaryClassification, scale, model.op.trainOptions.regClassification);
		value += scaleAndRegularize(wordVectorD, model.wordVectors, scale, model.op.trainOptions.regWordVector);
		value += scaleAndRegularize(wHiddenD, model.wHidden, scale, model.op.trainOptions.regTransform);

		derivative = RNNUtils.paramsToVector(theta.length, binaryTD.valueIterator(), 
				binaryCD.valueIterator(), 
				SimpleTensor.iteratorSimpleMatrix(binaryTensorTD.valueIterator()), 
				unaryCD.values().iterator(), 
				wHiddenD.valueIterator(),
				wordVectorD.values().iterator());
	}

	public void setAdditionalFeatures(Tree trainingTree, Integer verbIndex) {
		//addtional features : 
		//1. path length(to be encoded as binary in 4 dimensions);
		//2. dominates
		//3. category of left child : 2 dim(None for leaf node, null, non-null)
		//4. category of right child : 2 dim 

		int nNodes = trainingTree.size();
		int maxEncodeLen = 4; 
		String maxEncodeStr = "1111";
		int numFeatures = maxEncodeLen + 1 ;



		Tree verbNode = trainingTree.getLeaves().get(verbIndex);
		SimpleMatrix features = new SimpleMatrix(nNodes, numFeatures);


		int i =0;
		for(Tree tree:trainingTree){
			//set path length feature
			Integer pathLength = trainingTree.pathNodeToNode(tree, verbNode).size();
			String pathBStr = Integer.toBinaryString(pathLength);
			int pathEncodeLen = pathBStr.length();
			if(pathEncodeLen > maxEncodeLen)
				pathBStr = maxEncodeStr;						
			int k = maxEncodeLen-1; //start from last dimension			
			for(int j=pathBStr.length()-1; j>=0; j--){
				if(pathBStr.charAt(j) == '0')
					features.set(i,k, 0);
				else
					features.set(i,k, 1);
				k--;
			}

			//set dominates feature
			if(tree.dominates(verbNode))
				features.set(i, numFeatures-1, 1 );
			else
				features.set(i, numFeatures-1, 0 );




			i++;
		}
		//		//calculate mean
		//		double mean = 0;
		//		for(i=0; i< nNodes; i++){
		//			mean += features.get(i,0);
		//		}
		//		mean = mean/(nNodes);
		//		//calculate variance
		//		double variance = 0;
		//		for(i=0; i<nNodes; i++){
		//			variance += (mean-features.get(i,0))*(mean-features.get(i,0));
		//		}
		//		variance = variance/(nNodes);
		//		//calculate standard deviation
		//		double std = Math.sqrt(variance);
		//		
		//		//normalize features
		//		for(i=0; i<nNodes; i++){
		//			features.set(i,0, features.get(i,0)-mean);
		//			features.set(i,0, features.get(i,0)/std);
		//		}

		//set additional features for each node
		i = 0;
		for(Tree tree : trainingTree){
			//			SimpleMatrix thisNodeFeatures = new SimpleMatrix(numFeatures, 1);
			//			thisNodeFeatures.set(features.extractMatrix(i, i+1, 0, numFeatures));

			RNNCoreAnnotations.setAdditionalFeatures(tree, features.extractMatrix(i, i+1, 0, numFeatures).transpose());
			i++;
		}
	}

	public Tree findSubTree(Tree tree, Integer verbId, Integer wordId){
		Tree maxnode = null;
		List<Tree> t = tree.getLeaves();
		if(verbId == wordId)
			maxnode = t.get(verbId);
		else{

			try{
				List<Tree> verbtoroot = tree.pathNodeToNode(t.get(verbId), tree);
				List<Tree> wordtoroot = tree.pathNodeToNode(t.get(wordId), tree);

				for (int i=1; i< verbtoroot.size(); i++){  //start from a non-terminal node
					Tree verbpathnode = verbtoroot.get(i);
					for (int j=1; j<wordtoroot.size(); j++){
						Tree wordpathnode = wordtoroot.get(j);
						if(wordpathnode.equals(verbpathnode)){
							maxnode = wordpathnode;
							break;
						}				 	
					}
					if(maxnode != null)
						break;
				}
			}catch(Exception e){
				System.out.println("Leaves: "+ t.toString());
				e.printStackTrace();
			}
		}

		if(maxnode==null)
			maxnode = tree;
		//		RNNCoreAnnotations.setGoldClass(maxnode, trueLabel);
		return maxnode;
	}

	public Tree findSubTree(Tree root, Tree verbNode, Tree node){
		Tree topNode = null;
		List<Tree> verbtoroot = root.pathNodeToNode(verbNode, root);
		List<Tree> nodetoroot = root.pathNodeToNode(node, root);
		if((root == node))
			topNode = root;

		else{
			for (int i=1; i< verbtoroot.size(); i++){  //start from a non-terminal node
				Tree verbpathnode = verbtoroot.get(i);
				for (int j=1; j<nodetoroot.size(); j++){
					Tree wordpathnode = nodetoroot.get(j);
					if(wordpathnode.equals(verbpathnode)){
						topNode = wordpathnode;
						break;
					}				 	
				}
				if(topNode != null)
					break;
			}
		}

		return topNode;
	}

	public void attachLabelsToNodes(Tree trainingTree, List<Integer> trueLabels, Categories iobCategories, Categories roleCategories ) {
		//attach "O" tag for each node initially
		Iterator<Tree> treeIter = trainingTree.iterator();
		while(treeIter.hasNext()){
			Tree tree = treeIter.next();
			RNNCoreAnnotations.setGoldClass(tree, roleCategories.get("O"));			
		}

		HashMap<Integer, String> revMap = new HashMap<>();
		for(String key : iobCategories.keySet()){
			revMap.put(iobCategories.get(key), key);
		}
		for(int i=0; i< trueLabels.size(); i++){

			if(revMap.get(trueLabels.get(i)).startsWith("B-") ){
				int endPos = i+1;
				while((endPos < trueLabels.size()) && revMap.get(trueLabels.get(endPos)).startsWith("I-"))
					endPos += 1;

				Tree subtree = findSubTree(trainingTree, i, endPos-1);
				Iterator<Tree> it = subtree.iterator();
				while(it.hasNext()){
					Tree tree = it.next();
					RNNCoreAnnotations.setGoldClass(tree, roleCategories.get(revMap.get(trueLabels.get(i)).substring(2)));			
				}
			}			
		}

	}

	/**
	 * 
	 * Sets last two dimensions of leaf nodes as the distance from the word-to-tag node
	 * and predicate node
	 * @param trainingTree
	 * @param verbIndex
	 * @param wid
	 */
	public void setVerbNWordIndexFeatures(Tree trainingTree, Integer verbIndex, int wid) {
		//minimum distance = 0 and maximum distance = 10
		double maxDistance = 10.0;
		List<Tree> words = trainingTree.getLeaves();
		for(int i=0; i< words.size(); i++){
			String wordStr = words.get(i).label().value();
			SimpleMatrix nodeVector = model.getWordVector(wordStr);
			int row = nodeVector.numRows() - 2;
			double word2word = i - wid;
			double word2verb = i - verbIndex;
			if((Math.abs(word2word) > 10))
				word2word = maxDistance;
			if(Math.abs(word2verb) > 10)
				word2verb = maxDistance;

			word2word /= maxDistance;
			word2verb /= maxDistance;

			nodeVector.set(row, 0, word2word);
			nodeVector.set(row+1, 0, word2verb);
		}

	}
	
	/**
	 * 
	 * Sets last two dimensions of leaf nodes as the distance from
	 * predicate node
	 * @param trainingTree
	 * @param verbIndex
	 * 
	 */
	public void setVerbNWordIndexFeatures(Tree trainingTree, Integer verbIndex	) {
		//minimum distance = 0 and maximum distance = 10
		double maxDistance = 10.0;
		List<Tree> words = trainingTree.getLeaves();
		for(int i=0; i< words.size(); i++){
			String wordStr = words.get(i).label().value();
			SimpleMatrix nodeVector = model.getWordVector(wordStr);
			int row = nodeVector.numRows() - 2;
			//			double word2word = i - wid;
			double word2verb = i - verbIndex;
			//			if((Math.abs(word2word) > 10))
			//				word2word = maxDistance;
			if(Math.abs(word2verb) > 10)
				word2verb = maxDistance;

			//			word2word /= maxDistance;
			word2verb /= maxDistance;

			//			nodeVector.set(row, 0, word2word);
			nodeVector.set(row+1, 0, word2verb);
		}

	}


	double scaleAndRegularize(TwoDimensionalMap<String, String, SimpleMatrix> derivatives,
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

	double scaleAndRegularize(Map<String, SimpleMatrix> derivatives,
			Map<String, SimpleMatrix> currentMatrices,
			double scale,
			double regCost) {
		double cost = 0.0; // the regularization cost
		for (Map.Entry<String, SimpleMatrix> entry : currentMatrices.entrySet()) {
			SimpleMatrix D = derivatives.get(entry.getKey());
			D = D.scale(scale).plus(entry.getValue().scale(regCost));
			derivatives.put(entry.getKey(), D);
			cost += entry.getValue().elementMult(entry.getValue()).elementSum() * regCost / 2.0;
		}
		return cost;
	}

	double scaleAndRegularizeTensor(TwoDimensionalMap<String, String, SimpleTensor> derivatives,
			TwoDimensionalMap<String, String, SimpleTensor> currentMatrices,
			double scale,
			double regCost) {
		double cost = 0.0; // the regularization cost
		for (TwoDimensionalMap.Entry<String, String, SimpleTensor> entry : currentMatrices) {
			SimpleTensor D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
			D = D.scale(scale).plus(entry.getValue().scale(regCost));
			derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
			cost += entry.getValue().elementMult(entry.getValue()).elementSum() * regCost / 2.0;
		}
		return cost;
	}


	private void backpropDerivativesAndError(Tree tree, Tree verbNode,
			TwoDimensionalMap<String, String, SimpleMatrix> binaryTD,
			TwoDimensionalMap<String, String, SimpleMatrix> binaryCD,
			TwoDimensionalMap<String, String, SimpleTensor> binaryTensorTD,
			Map<String, SimpleMatrix> unaryCD,
			TwoDimensionalMap<String, String, SimpleMatrix> wHiddenD,
			Map<String, SimpleMatrix> wordVectorD) {
		SimpleMatrix delta = new SimpleMatrix(model.op.numHid, 1);
		backpropDerivativesAndError(tree, verbNode, unaryCD, wHiddenD, wordVectorD);
		backpropDerivativesAndError(tree, tree, verbNode, binaryTD, binaryTensorTD, wordVectorD, delta);
	}


	private void backpropDerivativesAndError(Tree root, Tree verbNode, 
			Map<String, SimpleMatrix> unaryCD,
			TwoDimensionalMap<String, String, SimpleMatrix> wHiddenD,
			Map<String, SimpleMatrix> wordVectorD){

		Iterator<Tree> itr = root.iterator();
		while(itr.hasNext()){
			
			Tree tree = itr.next();
			Tree topNode = findSubTree(root, verbNode, tree);
//			SimpleMatrix topNodeVector = RNNCoreAnnotations.getNodeVector(topNode);
//			SimpleMatrix verbNodeVector = RNNCoreAnnotations.getNodeVector(verbNode);
//			SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);		

//			SimpleMatrix catInput = RNNUtils.concatenateWithBias(nodeVector, topNodeVector, verbNodeVector);
			List<Integer> nodeIds = getCatInputNodes(root, verbNode, tree, model.op.windowSize);
			SimpleMatrix catInput = getCatInput(root, topNode, tree, nodeIds);

			String category = tree.label().value();
			category = model.basicCategory(category);
			SimpleMatrix wHidden = model.getWHidden("", "");
			SimpleMatrix classification = model.getUnaryClassification(category);


			//calculate prediction error at this node
			// Build a vector that looks like 0,0,1,0,0 with an indicator for the correct class
			SimpleMatrix goldLabel = new SimpleMatrix(model.numClasses, 1);
			int goldClass = RNNCoreAnnotations.getGoldClass(tree);
			goldLabel.set(goldClass, 1.0);
			double nodeWeight = model.op.trainOptions.getClassWeight(goldClass);
			SimpleMatrix predictions = RNNCoreAnnotations.getPredictions(tree);		
			double error = -(RNNUtils.elementwiseApplyLog(predictions).elementMult(goldLabel).elementSum());
			error = error * nodeWeight;
			RNNCoreAnnotations.setPredictionError(tree, error);

			//calculate output delta and update derivatives of unaryCD
			SimpleMatrix topDelta = predictions.minus(goldLabel).scale(nodeWeight);
			SimpleMatrix hiddenValues = wHidden.mult(catInput);
			hiddenValues = RNNUtils.elementwiseApplyTanh(hiddenValues);
			SimpleMatrix softmaxCD = topDelta.mult(RNNUtils.concatenateWithBias(hiddenValues).transpose());
			unaryCD.put(category, unaryCD.get(category).plus(softmaxCD));


			//calcualte deltaH and update wHiddenD
			SimpleMatrix hiddenValuesDerivative = RNNUtils.elementwiseApplyTanhDerivative(hiddenValues);
			SimpleMatrix deltaH = classification.transpose().mult(topDelta);
			deltaH = deltaH.extractMatrix(0, model.nHidden, 0, 1).elementMult(hiddenValuesDerivative);
			SimpleMatrix hiddenD = deltaH.mult((catInput).transpose());
			wHiddenD.put("", "", wHiddenD.get("", "").plus(hiddenD));

			//calculate delta down
			SimpleMatrix deltaFromClass = wHidden.transpose().mult(deltaH);		

			for (Tree node : tree){
				RNNCoreAnnotations.setNodeDelta(node, new SimpleMatrix(model.op.numHid,1));
			}
			//set node deltas of verbnode, wordnode and topnode
			setNodeDeltas(root, topNode, tree, nodeIds, deltaFromClass, wordVectorD);
			
//			//add deltas to verbnode
//			SimpleMatrix verbNodeDelta = RNNCoreAnnotations.getNodeDelta(verbNode);
//			if(verbNodeDelta == null)
//				verbNodeDelta =  new SimpleMatrix(model.op.numHid,1);
//			RNNCoreAnnotations.setNodeDelta(verbNode, verbNodeDelta.plus(deltaFromClass.extractMatrix(2*model.op.numHid, 3*model.op.numHid, 0, 1)));
//
//			//add this node's delta
//			SimpleMatrix thisNodeDelta = RNNCoreAnnotations.getNodeDelta(tree);
//			if(thisNodeDelta == null)
//				thisNodeDelta =  new SimpleMatrix(model.op.numHid,1);
//			RNNCoreAnnotations.setNodeDelta(tree, thisNodeDelta.plus(deltaFromClass.extractMatrix(0, model.op.numHid, 0, 1)));
//
//			//add delta to topnode	
//			SimpleMatrix topNodeDelta = RNNCoreAnnotations.getNodeDelta(topNode);
//			if(topNodeDelta == null)
//				topNodeDelta =  new SimpleMatrix(model.op.numHid,1);
//			RNNCoreAnnotations.setNodeDelta(topNode, topNodeDelta.plus(deltaFromClass.extractMatrix(model.op.numHid, 2*model.op.numHid, 0, 1)));
		}

	}

	private void backpropDerivativesAndError(Tree tree, Tree root, Tree verbNode, 
			TwoDimensionalMap<String, String, SimpleMatrix> binaryTD,
			//			TwoDimensionalMap<String, String, SimpleMatrix> binaryCD,
			TwoDimensionalMap<String, String, SimpleTensor> binaryTensorTD,
			Map<String, SimpleMatrix> wordVectorD,
			SimpleMatrix deltaUp) {

		SimpleMatrix deltaFromClass = null;
		if (tree.isLeaf()) { 
			String word = tree.label().value();
			word = model.getVocabWord(word);
			deltaFromClass = RNNCoreAnnotations.getNodeDelta(tree);//deltaFromClass.extractMatrix(0, model.op.numHid, 0, 1);//.elementMult(currentVectorDerivative);
			SimpleMatrix deltaFull = deltaFromClass.plus(deltaUp);
			wordVectorD.put(word, wordVectorD.get(word).plus(deltaFull.extractMatrix(0, model.op.numHid, 0, 1)));
		} else {
			// Otherwise, this must be a binary node
			String leftCategory = model.basicCategory(tree.children()[0].label().value());
			String rightCategory = model.basicCategory(tree.children()[1].label().value());

			deltaFromClass = RNNCoreAnnotations.getNodeDelta(tree);//deltaFromClass.extractMatrix(0, model.op.numHid, 0, 1).elementMult(currentVectorDerivative);
			SimpleMatrix currentVectorDerivative = RNNUtils.elementwiseApplyTanhDerivative(RNNCoreAnnotations.getNodeVector(tree));
			deltaFromClass = deltaFromClass.elementMult(currentVectorDerivative);
			SimpleMatrix deltaFull = deltaFromClass.plus(deltaUp);

			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector);
			SimpleMatrix W_df = deltaFull.mult(childrenVector.transpose());
			binaryTD.put(leftCategory, rightCategory, binaryTD.get(leftCategory, rightCategory).plus(W_df));
			SimpleMatrix deltaDown;
			if (model.op.useTensors) {
				SimpleTensor Wt_df = getTensorGradient(deltaFull, leftVector, rightVector);
				binaryTensorTD.put(leftCategory, rightCategory, binaryTensorTD.get(leftCategory, rightCategory).plus(Wt_df));
				deltaDown = computeTensorDeltaDown(deltaFull, leftVector, rightVector, model.getBinaryTransform(leftCategory, rightCategory), model.getBinaryTensor(leftCategory, rightCategory));
			} else {
				deltaDown = model.getBinaryTransform(leftCategory, rightCategory).transpose().mult(deltaFull);
			}

			SimpleMatrix leftDerivative = RNNUtils.elementwiseApplyTanhDerivative(leftVector);
			SimpleMatrix rightDerivative = RNNUtils.elementwiseApplyTanhDerivative(rightVector);
			SimpleMatrix leftDeltaDown = deltaDown.extractMatrix(0, deltaFull.numRows(), 0, 1);
			SimpleMatrix rightDeltaDown = deltaDown.extractMatrix(deltaFull.numRows(), deltaFull.numRows() * 2, 0, 1);
			backpropDerivativesAndError(tree.children()[0], root, verbNode, binaryTD,  binaryTensorTD,
					wordVectorD, leftDerivative.elementMult(leftDeltaDown));
			backpropDerivativesAndError(tree.children()[1], root, verbNode, binaryTD,  binaryTensorTD, 
					wordVectorD, rightDerivative.elementMult(rightDeltaDown));
		}
	}

	private SimpleMatrix computeTensorDeltaDown(SimpleMatrix deltaFull, SimpleMatrix leftVector, SimpleMatrix rightVector,
			SimpleMatrix W, SimpleTensor Wt) {
		SimpleMatrix WTDelta = W.transpose().mult(deltaFull);
		SimpleMatrix WTDeltaNoBias = WTDelta.extractMatrix(0, deltaFull.numRows() * 2, 0, 1);
		int size = deltaFull.getNumElements();
		SimpleMatrix deltaTensor = new SimpleMatrix(size*2, 1);
		SimpleMatrix fullVector = RNNUtils.concatenate(leftVector, rightVector);
		for (int slice = 0; slice < size; ++slice) {
			SimpleMatrix scaledFullVector = fullVector.scale(deltaFull.get(slice));
			deltaTensor = deltaTensor.plus(Wt.getSlice(slice).plus(Wt.getSlice(slice).transpose()).mult(scaledFullVector));
		}
		return deltaTensor.plus(WTDeltaNoBias);
	}

	private SimpleTensor getTensorGradient(SimpleMatrix deltaFull, SimpleMatrix leftVector, SimpleMatrix rightVector) {
		int size = deltaFull.getNumElements();
		SimpleTensor Wt_df = new SimpleTensor(size*2, size*2, size);
		// TODO: combine this concatenation with computeTensorDeltaDown?
		SimpleMatrix fullVector = RNNUtils.concatenate(leftVector, rightVector);
		for (int slice = 0; slice < size; ++slice) {
			Wt_df.setSlice(slice, fullVector.scale(deltaFull.get(slice)).mult(fullVector.transpose()));
		}
		return Wt_df;
	}

	/**
	 * This is the method to call for assigning  node vectors
	 * to the Tree.  After calling this, each of the non-leaf nodes will
	 * have the node vector.  The annotations filled in are
	 * the RNNCoreAnnotations.NodeVector. 
	 */
	public void forwardPropagateTree(Tree tree, Tree root, Tree verbNode) {
		SimpleMatrix nodeVector = null;
		//		SimpleMatrix raeVector = null;
		//		SimpleMatrix W1 = model.rae.getRAEW1();
		//		SimpleMatrix W2 = model.rae.getRAEW2();
		//		SimpleMatrix b1 = model.rae.getRAEb1();

		if (tree.isLeaf()) {			
			String word = tree.label().value();			
			nodeVector = model.getWordVector(word);	
			//			raeVector = model.rae.getRAEWordVector(word); //get the word vector associated with recusive auto-encoder

		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			forwardPropagateTree(tree.children()[0], root, verbNode);
			forwardPropagateTree(tree.children()[1], root, verbNode);

			String leftCategory = tree.children()[0].label().value();
			String rightCategory = tree.children()[1].label().value();
			SimpleMatrix W = model.getBinaryTransform(leftCategory, rightCategory);


			SimpleMatrix leftVector = RNNCoreAnnotations.getNodeVector(tree.children()[0]);
			SimpleMatrix rightVector = RNNCoreAnnotations.getNodeVector(tree.children()[1]);
			SimpleMatrix childrenVector = RNNUtils.concatenateWithBias(leftVector, rightVector);
			if (model.op.useTensors) {
				SimpleTensor tensor = model.getBinaryTensor(leftCategory, rightCategory);
				SimpleMatrix tensorIn = RNNUtils.concatenate(leftVector, rightVector);
				SimpleMatrix tensorOut = tensor.bilinearProducts(tensorIn);        
				nodeVector = RNNUtils.elementwiseApplyTanh(W.mult(childrenVector).plus(tensorOut));
			} else {
				nodeVector = RNNUtils.elementwiseApplyTanh(W.mult(childrenVector));
			}
			//p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten()) 
			//			SimpleMatrix leftRAEVector = RNNCoreAnnotations.getRAEVector(tree.children()[0]);
			//			SimpleMatrix rightRAEVector = RNNCoreAnnotations.getRAEVector(tree.children()[1]);
			//			SimpleMatrix t1 = W1.mult(leftRAEVector);
			//			SimpleMatrix t2 = W2.mult(rightRAEVector);
			//			raeVector = t1.plus(t2);
			//			raeVector = raeVector.plus(b1);
			//			raeVector = RNNUtils.elementwiseApplyTanh(raeVector);

		} else {
			throw new AssertionError("Tree not correctly binarized");
		}

		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);
		//		label.set(RNNCoreAnnotations.RAEVector.class, raeVector);

	}

	public void attachPredictions(Tree tree, Tree root, Tree verbNode){
//		SimpleMatrix nodeVector = null;
		SimpleMatrix classification = null;
		//		SimpleMatrix leftChildPred = new SimpleMatrix(2,1);;
		//		SimpleMatrix rightChildPred = new SimpleMatrix(2,1);;


		if(tree.isLeaf()){
//			String word = tree.label().value();
//			nodeVector = model.getWordVector(word);
			//			leftChildPred.set(0, 0, 0); leftChildPred.set(1,0,0);
			//			rightChildPred.set(0, 0, 0); rightChildPred.set(1,0,0);
		}
		else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			attachPredictions(tree.children()[0], root, verbNode);
			attachPredictions(tree.children()[1], root, verbNode);
//			nodeVector = RNNCoreAnnotations.getNodeVector(tree);

			//			Integer leftChildPredIndx = RNNCoreAnnotations.getPredictedClass(tree.children()[0]);
			//			Integer rightChildPredIndx = RNNCoreAnnotations.getPredictedClass(tree.children()[1]);
			//			if(leftChildPredIndx == roleCats.get('O')) {
			//				leftChildPred.set(0, 0, 0); leftChildPred.set(1,0, 1);
			//			}
			//			else{
			//				leftChildPred.set(0, 0, 1); leftChildPred.set(1,0,0);
			//			}
			//			if(rightChildPredIndx == roleCats.get('O'))	{
			//				rightChildPred.set(0, 0, 0); rightChildPred.set(1,0,1);
			//			}
			//			else{
			//				rightChildPred.set(0, 0, 1); rightChildPred.set(1,0,0);
			//			}

		}

		classification = model.getUnaryClassification("");
		SimpleMatrix wHidden = model.getWHidden("", "");

		Tree topNode = findSubTree(root, verbNode, tree);
//		SimpleMatrix topNodeVector = null;		
//		topNodeVector = RNNCoreAnnotations.getNodeVector(topNode);		
//		SimpleMatrix verbNodeVector = RNNCoreAnnotations.getNodeVector(verbNode);
		//		SimpleMatrix additionalFeatures = RNNCoreAnnotations.getAdditionalFeatures(tree);
		//		SimpleMatrix raeVector = RNNCoreAnnotations.getRAEVector(tree);
		//		SimpleMatrix beforeAfter = new SimpleMatrix(1,1);
		//		if()
		//		beforeAfter.set(0,0,1);
		//		SimpleMatrix predictions = RNNUtils.softmax(classification.mult(RNNUtils.concatenateWithBias(nodeVector, topNodeVector, verbNodeVector)));
		//		SimpleMatrix predictions = RNNUtils.softmax(classification.mult(RNNUtils.concatenateWithBias(nodeVector, verbNodeVector, additionalFeatures)));
		//		SimpleMatrix predictions = RNNUtils.softmax(classification.mult(RNNUtils.concatenateWithBias(nodeVector, topNodeVector, 
		//				verbNodeVector, raeVector, additionalFeatures, leftChildPred, rightChildPred)));

//		SimpleMatrix catInput = RNNUtils.concatenateWithBias(nodeVector, topNodeVector, verbNodeVector);
		List<Integer> nodeIds = getCatInputNodes(root, verbNode, tree, model.op.windowSize);
		SimpleMatrix catInput = getCatInput(root, topNode, tree, nodeIds);
		SimpleMatrix hiddenValues = wHidden.mult(catInput);
		hiddenValues = RNNUtils.elementwiseApplyTanh(hiddenValues);
		hiddenValues = RNNUtils.concatenateWithBias(hiddenValues);
		SimpleMatrix predictions = RNNUtils.softmax(classification.mult(hiddenValues));

		int index = getPredictedClass(predictions);
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.Predictions.class, predictions);
		label.set(RNNCoreAnnotations.PredictedClass.class, index);
	}
	
	
	/** This method adds indexes of all the nodes to be used as input for classification
	 * 	Since node indexes are indexed starting from 1, index of the top node = 0 and for padding node = -1**/
	public List<Integer> getCatInputNodes(Tree root, Tree verbNode, Tree phraseNode, int windowSize) {
		List<Integer> nodes = Generics.newArrayList();

		List<Tree> leaves = root.getLeaves();
		CoreLabel label = null;

		//top node  //for top node add id as -1
		nodes.add(topNodeId);
		
		//verb node		
		label = (CoreLabel)verbNode.label();
		int vid = label.get(IndexAnnotation.class) ;
		nodes.add(vid);		
		
		//phrase node
		nodes.add(phraseNodeId);
		

		List<Tree> subtreeLeaves = phraseNode.getLeaves();
		//left context of phraseNode
		Tree leftLeaf = subtreeLeaves.get(0);
		label = (CoreLabel) leftLeaf.label();
		int lwid = label.get(IndexAnnotation.class); // indexed from 1
		for(int i=1 ; i <= windowSize; i++){
			if(lwid-i  <= 0)
				nodes.add(paddingNodeId);
			else{
				label = (CoreLabel)leaves.get(lwid-i-1).label();
				nodes.add(label.get(IndexAnnotation.class));
				//				nodes.add(leaves.get(wid-i));
			}
		}		
		//right context of word to tag
		Tree rLeaf = subtreeLeaves.get(subtreeLeaves.size()-1);
		label = (CoreLabel) rLeaf.label();
		int rwid = label.get(IndexAnnotation.class) ;
		for(int i=1 ; i <= windowSize; i++){
			if(rwid+i >= leaves.size())
				nodes.add(paddingNodeId);
			else{
				label = (CoreLabel)leaves.get(rwid+i - 1).label();
				nodes.add(label.get(IndexAnnotation.class));
				//				nodes.add(leaves.get(wid+i));
			}

		}

		//left context of verb
		for(int i=1 ; i <= windowSize; i++){
			if(vid-i <= 0)
				nodes.add(paddingNodeId);
			else{
				label = (CoreLabel)leaves.get(vid-i -1).label();
				nodes.add(label.get(IndexAnnotation.class));
				//						nodes.add(leaves.get(wid-i));
			}
		}		
		//right context of verb
		for(int i=1 ; i <= windowSize; i++){
			if(vid+i >= leaves.size())
				nodes.add(paddingNodeId);
			else{
				label = (CoreLabel)leaves.get(vid+i - 1).label();
				nodes.add(label.get(IndexAnnotation.class));
				//						nodes.add(leaves.get(wid+i));
			}

		}

		return nodes;
	}
	
	private SimpleMatrix getCatInput(Tree root, Tree subtree, Tree phraseNode, List<Integer> nodeIds) {
		SimpleMatrix catInput = null;
		SimpleMatrix thisvector = null;
		List<Tree> leaves = root.getLeaves();
		for (Integer nodeId : nodeIds){
			if(nodeId==paddingNodeId) //add padding vector
				thisvector = model.getWordVector(RNTNModel.PADDING);
			else if(nodeId > topNodeId)
				thisvector = model.getWordVector(leaves.get(nodeId-1).label().value());
			else if(nodeId == topNodeId) //topnode
				thisvector = RNNCoreAnnotations.getNodeVector(subtree);
			else if(nodeId == phraseNodeId)
				thisvector = RNNCoreAnnotations.getNodeVector(phraseNode);

			if(catInput == null)
				catInput = thisvector;
			else
				catInput = RNNUtils.concatenate(catInput, thisvector);	
		}

		return RNNUtils.concatenateWithBias(catInput);

		//		return RNNUtils.concatenateWithBias(RNNCoreAnnotations.getNodeVector(nodes.get(0))); //only topnode vector
	}
	
	private void setNodeDeltas(Tree root, Tree topNode, Tree phraseNode, List<Integer> nodeIds, SimpleMatrix deltaFromClass, Map<String, SimpleMatrix> wordVectorD) {
		int i = 0;
		List<Tree> leaves = root.getLeaves();
		for(Integer nodeId : nodeIds){
			if(nodeId == topNodeId) {//topnode
				SimpleMatrix nodeDelta = RNNCoreAnnotations.getNodeDelta(topNode);
				RNNCoreAnnotations.setNodeDelta(topNode, nodeDelta.plus(deltaFromClass.extractMatrix((i)*model.op.numHid, (i+1)*model.op.numHid, 0, 1)));
			}else if(nodeId == paddingNodeId){ //add delta into padding word
				String word = model.getVocabWord(RNTNModel.PADDING);
				wordVectorD.put(word, wordVectorD.get(word).plus(deltaFromClass.extractMatrix((i)*model.op.numHid, (i+1)*model.op.numHid, 0, 1)));
			}else if(nodeId == phraseNodeId ){
				SimpleMatrix nodeDelta = RNNCoreAnnotations.getNodeDelta(phraseNode);
				RNNCoreAnnotations.setNodeDelta(phraseNode, nodeDelta.plus(deltaFromClass.extractMatrix((i)*model.op.numHid, (i+1)*model.op.numHid, 0, 1)));
			}else{
				if(nodeId > topNodeId){
					SimpleMatrix nodeDelta = RNNCoreAnnotations.getNodeDelta(leaves.get(nodeId-1));
					RNNCoreAnnotations.setNodeDelta(leaves.get(nodeId-1), nodeDelta.plus(deltaFromClass.extractMatrix((i)*model.op.numHid, (i+1)*model.op.numHid, 0, 1)));
				}
			}
			i++;
		}		
	}
	

}
