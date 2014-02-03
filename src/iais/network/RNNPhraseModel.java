package iais.network;

import java.io.Serializable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.rnn.RNNUtils;
import edu.stanford.nlp.rnn.SimpleTensor;
import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.TwoDimensionalMap;
import edu.stanford.nlp.util.TwoDimensionalSet;


/**
 * Model for learning Phrase Representations using Recursive Neural Network Architecture
 * 
 * @author bhanu
 *
 */

public class RNNPhraseModel implements Serializable {

	
	/**
	 * Nx2N+1, where N is the size of the word vectors
	 */
	public TwoDimensionalMap<String, String, SimpleMatrix> encodeTransform;	
	
	/**
	 * 2NxN+1, where N is the size of the word vectors
	 */
	public TwoDimensionalMap<String, String, SimpleMatrix> decodeTransform;	
	
	
	/**
	 * Scoring matrix for compatibility function
	 */
	public TwoDimensionalMap<String, String, SimpleMatrix> wScore;	
	
	/**
	 * Hidden layer weights for compatibility function
	 */
	public TwoDimensionalMap<String, String, SimpleMatrix> wHidden;	
	
	/**
	 * Size of the hidden layer used for compatibility function
	 */
	public final int nHidden = 100;

	
	/**
	 * Dimension of hidden layers, size of word vectors, etc
	 */
	public final int numHid;
	
	/**
	 * Alpha : Error = alpha*Ranking + (1-alpha)*ReconstructionCost
	 */
	public double alpha = 0.2;


	/** How many elements a transformation matrix has */
	public final int encodeTransformSize;
	public final int decodeTransformSize;
	public final int wScoreSize;
	public final int wHiddenSize;
	

	public Map<String, SimpleMatrix> wordVectors;
	
	/**
	 * we just keep this here for convenience
	 */
	transient SimpleMatrix identity;

	/** 
	 * A random number generator - keeping it here lets us reproduce results
	 */
	public final Random rand;
	public final int SEED = 42;
	public Random randContext ; //used when evaluating or checking gradient, to generate the same sequence

	public static final String UNKNOWN_WORD = "*UNK*";
	public static final String PADDING = "*PAD*";
	//  static final String RIGHT_PADDING = "*RPAD*";

	/**
	 * Will store various options specific to this model
	 */
	public final RNNPhraseOptions op;

	public boolean gradientCheck = false;

	private RNNPhraseModel(TwoDimensionalMap<String, String, SimpleMatrix> encodeTransform, 
			TwoDimensionalMap<String, String, SimpleMatrix> decodeTransform,
			TwoDimensionalMap<String, String, SimpleMatrix> wScore,
			TwoDimensionalMap<String, String, SimpleMatrix> wHidden,
			RNNPhraseOptions op) {
		this.op = op;
		
		this.encodeTransform = encodeTransform;
		this.decodeTransform = decodeTransform;
		this.wScore = wScore;
		this.wHidden = wHidden;
		if (op.numHid <= 0) {
			int nh = 0;
			for (SimpleMatrix wv : wordVectors.values()) {
				nh = wv.getNumElements();
			}
			this.numHid = nh;
		} else {
			this.numHid = op.numHid;		}
		
//		encodeTransformSize = numHid * ((2 + 2*op.nWordsInContext) * numHid + 1);
//		decodeTransformSize = (2 + 2*op.nWordsInContext)*numHid * (numHid + 1);
		encodeTransformSize = numHid * ((2 ) * numHid + 1);
		decodeTransformSize = (2)*numHid * (numHid + 1);
		wScoreSize = 1*((nHidden)+ 1);	
		wHiddenSize = (nHidden)*((2*op.nWordsInContext + 1)*(numHid) + 1);

		rand = new Random(op.randomSeed);

		identity = SimpleMatrix.identity(numHid);
	}

	/**
	 * The traditional way of initializing an empty model suitable for training.
	 */
	public RNNPhraseModel(RNNPhraseOptions op, List<Tree> trainingTrees) {
		this.op = op;
		
		rand = new Random(op.randomSeed);

		if (op.randomWordVectors) {
			initRandomWordVectors(trainingTrees);
		} else {
			readWordVectors();
		}
		if (op.numHid > 0) {
			this.numHid = op.numHid;
		} else {
			int size = 0;
			for (SimpleMatrix vector : wordVectors.values()) {
				size = vector.getNumElements();
				break;
			}
			this.numHid = size;
		}

		TwoDimensionalSet<String, String> binaryProductions = TwoDimensionalSet.hashSet();
		if (op.simplifiedModel) {
			binaryProductions.add("", "");
		} else {
			// TODO
			// figure out what binary productions we have in these trees
			// Note: the current sentiment training data does not actually
			// have any constituent labels
		}

		Set<String> unaryProductions = Generics.newHashSet();
		if (op.simplifiedModel) {
			unaryProductions.add("");
		} else {
			// TODO
			// figure out what unary productions we have in these trees (preterminals only, after the collapsing)
		}


		identity = SimpleMatrix.identity(numHid);

		encodeTransform = TwoDimensionalMap.treeMap();
		decodeTransform = TwoDimensionalMap.treeMap();
		wScore = TwoDimensionalMap.treeMap();
		wHidden = TwoDimensionalMap.treeMap();
		

		// When making a flat model (no symantic untying) the
		// basicCategory function will return the same basic category for
		// all labels, so all entries will map to the same matrix
		for (Pair<String, String> binary : binaryProductions) {
			String left = basicCategory(binary.first);
			String right = basicCategory(binary.second);
			if (encodeTransform.contains(left, right)) {
				continue;
			}			
			encodeTransform.put(left, right, randomEncodeMatrix());
			
			if(decodeTransform.contains(left,right)){
				continue;
			}
			decodeTransform.put(left,right, randomDecodeMatrix());
			
			if(wScore.contains(left,right)){
				continue;
			}
			wScore.put(left,right, SimpleMatrix.random(1, (nHidden + 1),-0.1, 0.1, rand));
			
			if(wHidden.contains(left,right)){
				continue;
			}
			wHidden.put(left,right, SimpleMatrix.random(nHidden, (2*op.nWordsInContext + 1)*(numHid)+1,-0.1, 0.1, rand));
			
		}
//		encodeTransformSize = numHid * ((2 + 2*op.nWordsInContext) * numHid + 1);
//		decodeTransformSize = (2 + 2*op.nWordsInContext)*numHid * (numHid + 1);
		encodeTransformSize = numHid * ((2 ) * numHid + 1);
		decodeTransformSize = (2)*numHid * (numHid + 1);
		wScoreSize = 1*(nHidden+ 1) ;
		wHiddenSize = (nHidden)*((2*op.nWordsInContext + 1)*(numHid) + 1);
		
		
	}

	

	SimpleMatrix randomEncodeMatrix() {
		SimpleMatrix binary = new SimpleMatrix(numHid, numHid * (2) + 1);
		// bias column values are initialized zero
		binary.insertIntoThis(0, 0, randomTransformBlock());
		binary.insertIntoThis(0, numHid, randomTransformBlock());
		return binary.scale(op.trainOptions.scalingForInit);
	}
	
	SimpleMatrix randomDecodeMatrix(){
		SimpleMatrix binary = new SimpleMatrix((2 )*numHid, numHid + 1);
		// bias column values are initialized zero
		binary.insertIntoThis(0, 0, randomTransformBlock());
		binary.insertIntoThis(numHid, 0, randomTransformBlock());
		return binary.scale(op.trainOptions.scalingForInit);
	}

	SimpleMatrix randomTransformBlock() {
//		    double range = 1.0 / (Math.sqrt((double) numHid) * 2.0);
//		    return SimpleMatrix.random(numHid,numHid,-range,range,rand).plus(identity);
		double range = 0.001;
		SimpleMatrix noise = SimpleMatrix.random(numHid, numHid,-range, range, rand);
		SimpleMatrix block = SimpleMatrix.identity(numHid).scale(0.5);
		return block.plus(noise);
	}

	SimpleMatrix randomWordVector() {
		return randomWordVector(op.numHid, rand);
	}

	static SimpleMatrix randomWordVector(int size, Random rand) {
		return RNNUtils.randomGaussian(size, 1, rand);
	}

	void initRandomWordVectors(List<Tree> trainingTrees) {
		if (op.numHid == 0) {
			throw new RuntimeException("Cannot create random word vectors for an unknown numHid");
		}
		Set<String> words = Generics.newHashSet();
		words.add(UNKNOWN_WORD);
		words.add(PADDING);
		for (Tree tree : trainingTrees) {
			List<Tree> leaves = tree.getLeaves();
			for (Tree leaf : leaves) {
				String word = leaf.label().value();
				if (op.lowercaseWordVectors) {
					word = word.toLowerCase();

				}
				if(word==null){
					System.out.println("got null word");
					continue;
				}
				words.add(word);
			}
		}
		this.wordVectors = Generics.newTreeMap();
		for (String word : words) {
			SimpleMatrix vector = randomWordVector();
			wordVectors.put(word, vector);
		}
	}

	void readWordVectors() {
		this.wordVectors = Generics.newTreeMap();
		Map<String, SimpleMatrix> rawWordVectors = RNNUtils.readRawWordVectors(op.wordVectors, op.numHid);
		for (String word : rawWordVectors.keySet()) {
			// TODO: factor out unknown word vector code from DVParser
			wordVectors.put(word, rawWordVectors.get(word));
		}

		String unkWord = op.unkWord;
		String padWord = op.padWord;
		SimpleMatrix unknownWordVector = wordVectors.get(unkWord);
		SimpleMatrix paddingWordVector = wordVectors.get(padWord);
		wordVectors.put(UNKNOWN_WORD, unknownWordVector);
		wordVectors.put(PADDING, paddingWordVector);
		if (unknownWordVector == null || paddingWordVector == null) {
			throw new RuntimeException("Unknown or Padding word vector not specified in the word vector file");
		}

	}

	public int totalParamSize() {
		int totalSize = 0;
		totalSize = (encodeTransformSize + decodeTransformSize + wScoreSize + wHiddenSize);
		return totalSize;
	}

	public double[] paramsToVector() {
		int totalSize = totalParamSize();
		return RNNUtils.paramsToVector(totalSize, encodeTransform.valueIterator(), decodeTransform.valueIterator(), wScore.valueIterator(), wHidden.valueIterator());
	}

	public void vectorToParams(double[] theta) {
		RNNUtils.vectorToParams(theta, encodeTransform.valueIterator(), decodeTransform.valueIterator(), wScore.valueIterator(), wHidden.valueIterator());
	}

	// TODO: combine this and getClassWForNode?
	public SimpleMatrix getWForNode(Tree node) {
		if (node.children().length == 2) {
			String leftLabel = node.children()[0].value();
			String leftBasic = basicCategory(leftLabel);
			String rightLabel = node.children()[1].value();
			String rightBasic = basicCategory(rightLabel);
			return encodeTransform.get(leftBasic, rightBasic);      
		} else if (node.children().length == 1) {
			throw new AssertionError("No unary transform matrices, only unary classification");
		} else {
			throw new AssertionError("Unexpected tree children size of " + node.children().length);
		}
	}

	public SimpleMatrix getWordVector(String word) {
		return wordVectors.get(getVocabWord(word));
	}

	public String getRandomVocabWord(){
		List<String> keys      = new ArrayList<String>(wordVectors.keySet());
		String       randomWord = keys.get( rand.nextInt(keys.size()) );
		return randomWord;
	}
	
	public String getRandomVocabWord(Random rand){
		List<String> keys      = new ArrayList<String>(wordVectors.keySet());
		String       randomWord = keys.get( rand.nextInt(keys.size()) );
		return randomWord;
	}
	
	public String getVocabWord(String word) {
		if (op.lowercaseWordVectors) {
			word = word.toLowerCase();
		}
		if (wordVectors.containsKey(word)) {
			return word;
		}
		
		// TODO: go through unknown words here
		return UNKNOWN_WORD;
	}

	public String basicCategory(String category) {
		if (op.simplifiedModel) {
			return "";
		}
		String basic = op.langpack.basicCategory(category);
		if (basic.length() > 0 && basic.charAt(0) == '@') {
			basic = basic.substring(1);
		}
		return basic;
	}

	

	public SimpleMatrix getEncodeTransform(String left, String right) {
		left = basicCategory(left);
		right = basicCategory(right);
		return encodeTransform.get(left, right);
	}
	
	public SimpleMatrix getDecodeTransform(String left, String right) {
		left = basicCategory(left);
		right = basicCategory(right);
		return decodeTransform.get(left, right);
	}

	public SimpleMatrix getWScore(String left, String right) {
		left = basicCategory(left);
		right = basicCategory(right);
		return wScore.get(left, right);
	}

	public SimpleMatrix getWHidden(String left, String right) {
		left = basicCategory(left);
		right = basicCategory(right);
		return wHidden.get(left, right);
	}
	
	public void saveSerialized(String path) {
		try {
			IOUtils.writeObjectToFile(this, path);
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}
	}

	public static RNNPhraseModel loadSerialized(String path) {
		try {
			return IOUtils.readObjectFromFile(path);
//			      return IOUtils.readObjectFromURLOrClasspathOrFileSystem(path);
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		} catch (ClassNotFoundException e) {
			throw new RuntimeIOException(e);
		}
	}

	public void printParamInformation(int index) {
		int curIndex = 0;
		for (TwoDimensionalMap.Entry<String, String, SimpleMatrix> entry : encodeTransform) {
			if (curIndex <= index && curIndex + entry.getValue().getNumElements() > index) {
				System.err.println("Index " + index + " is element " + (index - curIndex) + " of binaryTransform \"" + entry.getFirstKey() + "," + entry.getSecondKey() + "\"");
				return;
			} else {
				curIndex += entry.getValue().getNumElements();
			}
		}

		System.err.println("Index " + index + " is beyond the length of the parameters; total parameter space was " + totalParamSize());
	}

	private static final long serialVersionUID = 1;
}
