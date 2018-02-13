# Creates a sequence of features given feature files of ted corpus
from __future__ import print_function
from optparse import OptionParser
import os
import sys
import re
from pandas import DataFrame, read_csv
import csv
import cPickle
from random import randint
from collections import OrderedDict
import numpy as np
from utilities import sylco

csv.field_size_limit(1000000000000) 

features_f0_header = 'mean.normF0\tsd.normF0\tmax.normF0\tmin.normF0\tmedian.normF0\tq1.normF0\tq2.5.normF0\tq5.normF0\tq25.normF0\tq75.normF0\tq95.normF0\tq97.5.normF0\tq99.normF0\tslope.normF0\tintercept.normF0\tmean.normF0.slope\tsd.normF0.slope\tmax.normF0.slope\tmin.normF0.slope\tmedian.normF0.slope\tq1.normF0.slope\tq2.5.normF0.slope\tq5.normF0.slope\tq25.normF0.slope\tq75.normF0.slope\tq95.normF0.slope\tq97.5.normF0.slope\tq99.normF0.slope\tslope.normF0.slope\tintercept.normF0.slope'
features_i0_header = 'mean.normI0\tsd.normI0\tmax.normI0\tmin.normI0\tmedian.normI0\tq1.normI0\tq2.5.normI0\tq5.normI0\tq25.normI0\tq75.normI0\tq95.normI0\tq97.5.normI0\tq99.normI0\tslope.normI0\tintercept.normI0\tmean.normI0.slope\tsd.normI0.slope\tmax.normI0.slope\tmin.normI0.slope\tmedian.normI0.slope\tq1.normI0.slope\tq2.5.normI0.slope\tq5.normI0.slope\tq25.normI0.slope\tq75.normI0.slope\tq95.normI0.slope\tq97.5.normI0.slope\tq99.normI0.slope\tslope.normI0.slope\tintercept.normI0.slope'

#PUNC_DICT = [",", '.', '?', '!',':', ';', '-', '']
SPACE = "_"
PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?', 4:'!', 5:'-', 6:';', 7:':'}
INV_PUNCTUATION_CODES = {SPACE:0, ',':1, '.':2, '?':3, '!':4, '-':5, ';':6, ':':7, '':0}
REDUCED_PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?'}
REDUCED_INV_PUNCTUATION_CODES = {SPACE:0, ',':1, '.':2, '?':3, '':0}
EOS_PUNCTUATION_CODES = [2,3,4,5,6,7]
EOS_PUNCTUATION = ['.', '?', '!', '-', ':']

FLOAT_FORMATTING="{0:.4f}"

def puncProper(punc):
	if punc in INV_PUNCTUATION_CODES.keys():
		return punc
	else:
		return puncEstimate(punc)

def reducePuncCode(puncCode):
	if puncCode in [4, 5, 6, 7]: #period
		return 2
	else:
		return puncCode

def reducePunc(punc):
	puncCode = INV_PUNCTUATION_CODES[punc]
	reducedPuncCode = reducePuncCode(puncCode)
	return PUNCTUATION_VOCABULARY[reducedPuncCode]

def puncEstimate(punc):
	if '.' in punc:
		return '.'
	elif ',' in punc:
		return ','
	elif '?' in punc:
		return '?'
	elif '!' in punc:
		return '!'
	elif ':' in punc:
		return ':'
	elif ';' in punc:
		return ';'
	elif '-' in punc:
		return '-'
	else:
		return ''

def checkFile(filename, variable):
    if not filename:
        print("%s file not given"%variable)
        sys.exit()
    else:
        if not os.path.isfile(filename):
            print("%s file %s does not exist"%(variable, filename))
            sys.exit()

def readTedDataToMemory(file_word, file_wordalign, file_wordaggs_f0, file_wordaggs_i0):
	#read word file to a dictionary (word.txt)
	word_data_simple_dic = OrderedDict()
	with open(options.file_word, 'rt') as f:
		reader = csv.reader(f, delimiter='\t', quotechar=None)
		for row in reader:
			word_data_simple_dic[row[1]] = [row[12],row[9],row[10], row[14]]   #word, starttime, endtime, punctuation

	#read wordaggs_f0 file to a dictionary 
	word_id_to_f0_features_dic = {}
	with open(options.file_wordaggs_f0, 'rt') as f:
		reader = csv.reader(f, delimiter=' ', quotechar=None)
		for row in reader:
			word_id_to_f0_features_dic[row[0]] = row[6:36]

	#read wordaggs_i0 file to a dictionary
	word_id_to_i0_features_dic = {}
	with open(options.file_wordaggs_i0, 'rt') as f:
		reader = csv.reader(f, delimiter=' ', quotechar=None)
		for row in reader:
			word_id_to_i0_features_dic[row[0]] = row[6:36]

	#read aligned word file to a dictionary (word.align)
	word_data_aligned_dic = OrderedDict()
	with open(options.file_wordalign, 'rt') as f:
		reader = csv.reader(f, delimiter='\t', quotechar=None)
		first_line = 1
		for row in reader:
			if first_line:
				first_line = 0
				continue
			if not row[0] == "-":
				try:
					word_data_aligned_dic[row[7]] += [[row[5], row[6], row[1], row[3], row[4], row[19]]] #word.id (detailed), sent.id, word.stripped, starttime, endtime, punctuation
				except Exception, e:
					word_data_aligned_dic[row[7]] = [[row[5], row[6], row[1], row[3], row[4], row[19]]]

	return [word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic]

def featureVectorToFloat(featureVector):
	features_fixed = [0.0] * len(featureVector)
	for ind, val in enumerate(featureVector):
		if val == 'NA':
			features_fixed[ind] = 0.0
		else:
			features_fixed[ind] = float(FLOAT_FORMATTING.format(float(val)))
	return features_fixed

def structureData(word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic):
	structured_data = []
	speech_rate_vals = []
	non_zero_pause_vals = []

	prev_wordEntry = {'starttime':0.0, 'endtime':0.0, 'punc_before':"", 'punc_after':"", 
					  'features_f0':[0] * 29,
					  'features_i0':[0] * 29}
	for key in word_data_aligned_dic:
		#case of it's that's
		if len(word_data_aligned_dic[key]) == 2 and re.search(r"^{", word_data_aligned_dic[key][1][2]):
			word_data_aligned_dic[key][0][2] += "'" + word_data_aligned_dic[key][1][2][1:]
			word_data_aligned_dic[key][0][4] = word_data_aligned_dic[key][1][4]
			del word_data_aligned_dic[key][1]

		for word_index, word_data in enumerate(word_data_aligned_dic[key]):
			wordEntry = {'sent.id':"", 'word.id':"", 'word.id.simple':"", 'word':"", 
					     'word.stripped':"", 'utt_pos':"", 'punc_before':"", 'punc_after':"", 'total_punc_before':"",
					     'minimal_punc_before': "", 'starttime':0.0, 'endtime':0.0, 'starttime.approx':0, 
					     'endtime.approx':0, 'features_f0':[0], 'pause_before_dur':0.0, 
					     'features_i0':[0], 'mean.f0_jump_from_prev':0.0, 'mean.i0_jump_from_prev':0.0,
					     'range.f0':0.0, 'range.i0':0.0, 'word_dur':0.0,
					     'speech.rate':0.0}
			wordEntry['word.id.simple'] = key
			wordEntry['word.id'] = word_data[0]
			wordEntry['sent.id'] = word_data[1]
			word_stripped = word_data[2]

			if not word_data[3] == "NA": 
				wordEntry['starttime'] = float(word_data[3])
			else:
				wordEntry['starttime'] = -1
			if not word_data[4] == "NA": 
				wordEntry['endtime'] = float(word_data[4])
			else:
				wordEntry['endtime'] = -1

			if re.search(r"\w", word_stripped) == None:
				continue

			#strip word from non-word stuff at the beginning and end
			word_stripped = word_stripped[re.search(r"\w", word_stripped).start():]
			word_stripped = word_stripped[::-1]
			word_stripped = word_stripped[re.search(r"\w", word_stripped).start():]
			word_stripped = word_stripped[::-1]

			wordEntry['word.stripped'] += word_stripped
			
			#pull info from other files
			try:
				wordEntry['word'] = word_data_simple_dic[wordEntry['word.id.simple']][0]
				wordEntry['starttime.approx'] = word_data_simple_dic[wordEntry['word.id.simple']][1]
				wordEntry['endtime.approx'] = word_data_simple_dic[wordEntry['word.id.simple']][2]
			except Exception, e:
				print("problem with other files for %s"%(wordEntry['word.id']))

			#skip speaker turn information which is denoted as {xx} in word.txt
			if re.search(r"{|}", wordEntry['word']):
				continue

			#sometimes word file has the word transcription wrong. take care of those cases
			if re.search(r"\w", wordEntry['word']) == None:
				wordEntry['word'] = wordEntry['word.stripped']

			try:
				wordEntry['features_f0'] = word_id_to_f0_features_dic[wordEntry['word.id']]
			except Exception as e:
				wordEntry['features_f0'] = [0] * 29
			
			try:
				wordEntry['features_i0'] = word_id_to_i0_features_dic[wordEntry['word.id']]
			except Exception as e:
				wordEntry['features_i0'] = [0] * 29
			
			#pause values
			if not wordEntry['starttime'] == -1 and not prev_wordEntry['endtime'] == -1:
				diff = wordEntry['starttime'] - prev_wordEntry['endtime']
			else:
				diff = 0.0
			wordEntry['pause_before_dur'] = float(FLOAT_FORMATTING.format(diff))

			#word duration
			if not wordEntry['starttime'] == -1 and not prev_wordEntry['endtime'] == -1:
				diff = wordEntry['endtime'] - wordEntry['starttime']
			else:
				diff = 0.0
			wordEntry['word_dur'] = float(FLOAT_FORMATTING.format(diff))

			#speech rate with respect to syllables
			no_syllables = float(sylco(wordEntry['word.stripped']))

			#speech rate with respect to phonemes (no of characters)
			no_of_characters = len(re.sub('[^a-zA-Z]','',wordEntry['word.stripped']))
			sr_unit = no_syllables

			if wordEntry['word_dur'] > 0:
				speech_rate = sr_unit / wordEntry['word_dur']
			else:
				speech_rate = 0.0
			wordEntry['speech.rate'] = float(FLOAT_FORMATTING.format(speech_rate))

			#print(wordEntry['word.stripped'])
			#print("no_syllables: %s, dur:%s, speech.rate: %s,"%(no_syllables, wordEntry['word_dur'], wordEntry['speech.rate']))
			#print("---")

			if wordEntry['speech.rate'] > 0:
				speech_rate_vals.append(wordEntry['speech.rate'])
			if wordEntry['pause_before_dur'] > 0:
				non_zero_pause_vals.append(wordEntry['pause_before_dur'])
			
			#convert i0 and f0 feature vectors to float vectors
			wordEntry['features_f0'] = featureVectorToFloat(wordEntry['features_f0'])
			wordEntry['features_i0'] = featureVectorToFloat(wordEntry['features_i0'])

			#other prosodic features
			#jump.f0 = mean.f0 of the current word - mean.f0 of the previous word
			f0_jump = wordEntry['features_f0'][0] - prev_wordEntry['features_f0'][0]
			wordEntry['mean.f0_jump_from_prev'] = float(FLOAT_FORMATTING.format(f0_jump))
			#jump.i0 = mean.i0 of the current word - mean.i0 of the previous word
			i0_jump = wordEntry['features_i0'][0] - prev_wordEntry['features_i0'][0]
			wordEntry['mean.i0_jump_from_prev'] = float(FLOAT_FORMATTING.format(i0_jump))
			#range.f0 = max.f0 - min.f0
			f0_range = wordEntry['features_f0'][2] - wordEntry['features_f0'][3]
			wordEntry['range.f0'] = float(FLOAT_FORMATTING.format(f0_range))
			#range.i0 = max.i0 - min.i0
			i0_range = wordEntry['features_i0'][2] - wordEntry['features_i0'][3]
			wordEntry['range.i0'] = float(FLOAT_FORMATTING.format(i0_range))
			#birange.f0 = max(max_f0 of current and previous word) - min(min_f0 of current and previous word)
			f0_bigram_max = max(wordEntry['features_f0'][2], prev_wordEntry['features_f0'][2])
			f0_bigram_min = min(wordEntry['features_f0'][3], prev_wordEntry['features_f0'][3])
			f0_birange = f0_bigram_max - f0_bigram_min
			wordEntry['birange.f0'] = float(FLOAT_FORMATTING.format(f0_birange))
			#birange.i0 = max(max_i0 of current and previous word) - min(min_i0 of current and previous word)
			i0_bigram_max = max(wordEntry['features_i0'][2], prev_wordEntry['features_i0'][2])
			i0_bigram_min = min(wordEntry['features_i0'][3], prev_wordEntry['features_i0'][3])
			i0_birange = i0_bigram_max - i0_bigram_min
			wordEntry['birange.i0'] = float(FLOAT_FORMATTING.format(i0_birange))
			
			#check punctuation marks
			word_being_processed = wordEntry['word']
			punc_after = ""
			punc_before = ""

			#check beginning
			if re.search(r"^\W", word_being_processed) and word_index == 0:
				punc = word_being_processed[:re.search(r"\w", word_being_processed).start()]
				punc_before += punc
				word_being_processed = word_being_processed[re.search(r"\w", word_being_processed).start():]

			#check end again (issue with quotations)
			word_reversed = word_being_processed[::-1]
			if re.search(r"^\W",word_reversed) and word_index == len(word_data_aligned_dic[key]) - 1:
				punc = word_reversed[:re.search(r"\w", word_reversed).start()][::-1]
				punc_after = punc + punc_after
				word_being_processed = word_reversed[re.search(r"\w", word_reversed).start():][::-1]

			wordEntry['punc_before'] = punc_before
			wordEntry['punc_after'] = punc_after

			total_punc_before = prev_wordEntry['punc_after'] + wordEntry['punc_before']

			wordEntry['total_punc_before'] = total_punc_before
			wordEntry['minimal_punc_before'] = puncProper(total_punc_before)

			structured_data += [wordEntry]
			prev_wordEntry = wordEntry

	return structured_data, speech_rate_vals, non_zero_pause_vals

def word_data_to_pickle(talk_data, output_pickle_file):
	with open(output_pickle_file, 'wb') as f:
		cPickle.dump(talk_data, f, cPickle.HIGHEST_PROTOCOL)

def word_data_to_csv(talk_data, output_csv_file):
	with open(output_csv_file, 'wb') as f:
		w = csv.writer(f, delimiter="|")
		rowIds = ['word', 'punctuation_before', 'pause_before', 'f0_mean', 'f0_range', 'f0_birange', 'f0_sd', 'i0_mean', 'i0_range', 'i0_birange',  'i0_sd', 'speech_rate_norm']
		w.writerow(rowIds)
		rows = zip( talk_data['word'],
					talk_data['punctuation'],
					talk_data['pause'],
					talk_data['mean.f0'],
					talk_data['range.f0'],
					talk_data['birange.f0'],
					talk_data['sd.f0'],
					talk_data['mean.i0'],
					talk_data['range.i0'],
					talk_data['birange.i0'],
					talk_data['sd.i0'],
					talk_data['speech.rate.norm'])
		for row in rows:                                        
			w.writerow(row) 

def wordDataToDictionary(structured_word_data, mean_speech_rate, std_speech_rate, mean_pause, std_pause):
	actualword_seq = []
	#speech_rate_syll_seq = []
	speech_rate_seq = []
	speech_rate_normalized_seq = []
	word_dur_seq = []
	punc_seq = []
	punc_reduced_seq = []
	pause_before_seq = []
	pause_before_dur_normalized_seq = []
	meanf0_seq = []
	medf0_seq = []
	meani0_seq = []
	slopef0_seq = []
	sdf0_seq = []
	sdi0_seq = []
	jumpf0_seq = []
	jumpi0_seq = []
	rangef0_seq = []
	rangei0_seq = []
	birangef0_seq = []
	birangei0_seq = []
	#id sequences
	meanf0_id_seq = []
	meani0_id_seq = []
	rangef0_id_seq = []
	rangei0_id_seq = []
	pause_id_seq = []
	punctuation_id_seq = []
	reduced_punctuation_id_seq = []

	for word_datum in structured_word_data:
		actualword_seq += [word_datum['word.stripped']]
		word_dur_seq += [word_datum['word_dur']]
		punc_seq += [word_datum['minimal_punc_before']]
		punc_reduced_seq += [reducePunc(word_datum['minimal_punc_before'])]
		pause_before_seq += [word_datum['pause_before_dur']]
		meanf0_seq += [word_datum['features_f0'][0]]
		meani0_seq += [word_datum['features_i0'][0]]
		sdf0_seq += [word_datum['features_f0'][1]]
		sdi0_seq += [word_datum['features_i0'][1]]
		medf0_seq += [word_datum['features_f0'][4]]
		slopef0_seq += [word_datum['features_f0'][14]]
		jumpf0_seq += [word_datum['mean.f0_jump_from_prev']]
		jumpi0_seq += [word_datum['mean.i0_jump_from_prev']]
		rangef0_seq += [word_datum['range.f0']]
		rangei0_seq += [word_datum['range.i0']]
		birangef0_seq += [word_datum['birange.f0']]
		birangei0_seq += [word_datum['birange.i0']]

		#punctuation
		punctuation_id = INV_PUNCTUATION_CODES[word_datum['minimal_punc_before']]
		punctuation_id_seq += [punctuation_id]
		reduced_punctuation_id_seq += [reducePuncCode(punctuation_id)]
		#speech rate
		speech_rate_seq += [word_datum['speech.rate']]
		
		if not word_datum['speech.rate'] == 0.0:
			normalized_speech_rate = (word_datum['speech.rate'] - mean_speech_rate) / std_speech_rate
		else:
			normalized_speech_rate = 0.0
		speech_rate_normalized_seq += [float(FLOAT_FORMATTING.format(normalized_speech_rate))]

		if not word_datum['pause_before_dur'] == 0.0:
			normalized_pause_before_dur = (word_datum['pause_before_dur'] - mean_pause) / std_pause
		else:
			normalized_pause_before_dur = 0.0
		pause_before_dur_normalized_seq += [float(FLOAT_FORMATTING.format(normalized_pause_before_dur))]


	metadata = {'no_of_words': len(actualword_seq)
	}

	talk_data = {  'word': actualword_seq,
				   'word.duration': word_dur_seq,
				   'speech.rate': speech_rate_seq,
				   'speech.rate.norm': speech_rate_normalized_seq,
				   'punctuation': punc_seq,
				   'punctuation.reduced': punc_reduced_seq,
				   'pause': pause_before_seq,
				   'pause.norm':pause_before_dur_normalized_seq,
				   'mean.f0': meanf0_seq,
				   'mean.i0': meani0_seq,
				   'med.f0': medf0_seq,
				   'slope.f0': slopef0_seq,
				   'sd.f0': sdf0_seq,
				   'sd.i0': sdi0_seq,
				   'jump.f0': jumpf0_seq,
				   'jump.i0': jumpi0_seq,
				   'range.f0': rangef0_seq,
				   'range.i0': rangei0_seq,
				   'birange.f0': birangef0_seq,
				   'birange.i0': birangei0_seq,
				   'punc.id': punctuation_id_seq,
				   'punc.red.id': reduced_punctuation_id_seq,
				   'metadata': metadata
	}
	return talk_data

def mean_std_calculator(values):
	return np.mean(values), np.std(values)

def main(options):
	#checkFile(options.file_audio, "file_audio")
	checkFile(options.file_word, "file_word")
	checkFile(options.file_wordalign, "file_wordalign")
	checkFile(options.file_wordaggs_f0, "file_wordaggs_f0")
	checkFile(options.file_wordaggs_i0, "file_wordaggs_i0")

	[word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic] = readTedDataToMemory(options.file_word, options.file_wordalign, options.file_wordaggs_f0, options.file_wordaggs_i0)
	[structured_word_data, speech_rate_vals, non_zero_pause_vals] = structureData(word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic)

	[mean_sr, sd_sr] = mean_std_calculator(speech_rate_vals)
	[mean_pause, sd_pause] = mean_std_calculator(non_zero_pause_vals)

	talk_data = wordDataToDictionary(structured_word_data, mean_sr, sd_sr, mean_pause, sd_pause)

	word_data_to_csv(talk_data, options.file_output_csv)

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-a", "--audio", dest="file_audio", default=None, help="wav", type="string")
	parser.add_option("-w", "--word", dest="file_word", default=None, help="word.txt", type="string")	#in /txt-sent
	parser.add_option("-l", "--align", dest="file_wordalign", default=None, help="word.txt.norm.align", type="string")	#in /txt-sent
	parser.add_option("-f", "--aggs_f0", dest="file_wordaggs_f0", default=None, help="aggs.alignword.txt under /derived/segs/f0/", type="string")	#in /derived/segs/f0/
	parser.add_option("-i", "--aggs_i0", dest="file_wordaggs_i0", default=None, help="aggs.alignword.txt under /derived/segs/i0/", type="string")	#in /derived/segs/i0/
	parser.add_option("-o", "--out_csv", dest="file_output_csv", default=None, help="outputfile (csv)", type="string")

	(options, args) = parser.parse_args()
	main(options)