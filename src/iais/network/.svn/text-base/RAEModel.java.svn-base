package iais.network;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;

import iais.io.Config;
import iais.io.MatFile;

import org.ejml.simple.SimpleMatrix;


import edu.stanford.nlp.io.IOUtils;

public class RAEModel implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1;
	private static final String PARAMS_MAT_FILE = Config.PROJECT_HOME+"data/corpus/params_rae.mat";
	private static final String RAE_WEMAT_FILE = Config.PROJECT_HOME+"data/corpus/vars.normalized.100.mat";
	private static final String RAE_WORDS_FILE = Config.PROJECT_HOME+"data/corpus/rae_words.txt";
	private static final String UNKNOWN_WORD = "*UNKNOWN*";
//	private MatFile paramsMat;
//	private MatFile weMat;
	private SimpleMatrix we, w1, w2, b1;
	private HashMap<String, Integer>	words;
	
	
	
	public RAEModel(){
		try {
			MatFile paramsMat = new MatFile(PARAMS_MAT_FILE);
			MatFile weMat = new MatFile(RAE_WEMAT_FILE); 
			
			w1 = paramsMat.readVar("W1");
			w2 = paramsMat.readVar("W2");
			b1 = paramsMat.readVar("b1");
			
			we = weMat.readVar("We");
			words = readWords();

			
			
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public SimpleMatrix getRAEWordVector(String word) {		
		Integer index = words.get(word);
		if(index==null)
			word = UNKNOWN_WORD;
		
		return we.extractMatrix(0, we.numRows(), words.get(word), words.get(word)+1);
	}

	public SimpleMatrix getRAEW1() {		
		return w1;
	}

	public SimpleMatrix getRAEW2() {		
		return w2;
	}

	public SimpleMatrix getRAEb1() {		
		return b1;
	}

	private HashMap<String, Integer> readWords(){		
		HashMap<String, Integer> words = new HashMap<>();
		int i = 0;
		for(String key : IOUtils.readLines(RAE_WORDS_FILE)){
			words.put(key.trim(), i);
			i++;
		}
		return words;
	}
	
}
