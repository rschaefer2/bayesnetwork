import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	
	private int v;
	private int spam_label_count = 0, ham_label_count = 0;
	private int spam_word_total = 0, ham_word_total = 0;
	private HashMap<String, Integer> spam_word_counts = new HashMap<String, Integer>();
	private HashMap<String, Integer> ham_word_counts = new HashMap<String, Integer>();
	
	@Override
	public void train(Instance[] trainingData, int v) {
		this.v = v;
		
		
		
		//count each word with label
		for(Instance inst: trainingData) {
			if( inst.label == Label.HAM) {
				ham_label_count++;
			} else if ( inst.label == Label.SPAM) {
				spam_label_count++;
			}
			for(String w: inst.words) {
				if( inst.label == Label.HAM) {	
					ham_word_total++;
					Integer count = ham_word_counts.get(w);
					if( count == null) {
						count = 0;
					}
					ham_word_counts.put(w, count+1);
				} else if( inst.label == Label.SPAM) {	
					spam_word_total++;
					Integer count = spam_word_counts.get(w);
					if( count == null) {
						count = 0;
					}
					spam_word_counts.put(w, count+1);
				}
			}
		}
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		if( label == Label.HAM)
			return (1.0*ham_label_count)/(1.0*ham_label_count+1.0*spam_label_count);
		else
			return (1.0*spam_label_count)/(1.0*ham_label_count+1.0*spam_label_count);
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		int count;
		HashMap<String, Integer> map;
		Integer w_count;
		if( label == Label.HAM) {
			map = this.ham_word_counts;
			count = this.ham_word_total;
		} else {
			map = this.spam_word_counts;
			count = this.spam_word_total;
		}
		
		w_count = map.get(word);
		if ( w_count == null) {
			w_count = 0;
		}
		
		return (w_count + 0.00001) / (this.v*0.00001 + count);
	}
	
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		double spam_prob = Math.log(p_l(Label.SPAM));
		double ham_prob = Math.log(p_l(Label.HAM));

		for(String w: words) {
			spam_prob += Math.log(p_w_given_l(w, Label.SPAM));
			ham_prob += Math.log(p_w_given_l(w, Label.HAM));
		}
		
		ClassifyResult cl = new ClassifyResult();
		cl.log_prob_ham = ham_prob;
		cl.log_prob_spam = spam_prob;
		cl.label = ham_prob > spam_prob ? Label.HAM : Label.SPAM;
		return cl;
	}
}
