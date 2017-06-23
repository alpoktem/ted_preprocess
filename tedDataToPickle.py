# Creates a sequence of features given feature files of ted corpus

from __future__ import print_function
from optparse import OptionParser
import os
import sys
import re
from pandas import DataFrame, read_csv
import pandas as pd 
import csv
import cPickle
from random import randint
from collections import OrderedDict

csv.field_size_limit(1000000000000) 

features_f0_header = 'mean.normF0\tsd.normF0\tmax.normF0\tmin.normF0\tmedian.normF0\tq1.normF0\tq2.5.normF0\tq5.normF0\tq25.normF0\tq75.normF0\tq95.normF0\tq97.5.normF0\tq99.normF0\tslope.normF0\tintercept.normF0\tmean.normF0.slope\tsd.normF0.slope\tmax.normF0.slope\tmin.normF0.slope\tmedian.normF0.slope\tq1.normF0.slope\tq2.5.normF0.slope\tq5.normF0.slope\tq25.normF0.slope\tq75.normF0.slope\tq95.normF0.slope\tq97.5.normF0.slope\tq99.normF0.slope\tslope.normF0.slope\tintercept.normF0.slope'
features_i0_header = 'mean.normI0\tsd.normI0\tmax.normI0\tmin.normI0\tmedian.normI0\tq1.normI0\tq2.5.normI0\tq5.normI0\tq25.normI0\tq75.normI0\tq95.normI0\tq97.5.normI0\tq99.normI0\tslope.normI0\tintercept.normI0\tmean.normI0.slope\tsd.normI0.slope\tmax.normI0.slope\tmin.normI0.slope\tmedian.normI0.slope\tq1.normI0.slope\tq2.5.normI0.slope\tq5.normI0.slope\tq25.normI0.slope\tq75.normI0.slope\tq95.normI0.slope\tq97.5.normI0.slope\tq99.normI0.slope\tslope.normI0.slope\tintercept.normI0.slope'
	
def puncProper(punc):
	if punc in punc_dict_minimal:
		return punc
	else:
		return puncEstimate(punc)

punc_dict_minimal = [",", '.', '?', '!',':', ';', '-','']

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
			features_fixed[ind] = float(val)
	return features_fixed

def structureData(word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic):
	structured_data = []

	prev_wordEntry = {'starttime':0, 'endtime':0, 'punc_before':"", 'punc_after':"", 
					  'features_f0':['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'],
					  'features_i0':['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']}
	for key in word_data_aligned_dic:
		#case of it's that's
		if len(word_data_aligned_dic[key]) == 2 and re.search(r"^{", word_data_aligned_dic[key][1][2]):
			word_data_aligned_dic[key][0][2] += "'" + word_data_aligned_dic[key][1][2][1:]
			word_data_aligned_dic[key][0][4] = word_data_aligned_dic[key][1][4]
			del word_data_aligned_dic[key][1]

		#check if word.stripped differs from word
		word_in_pieces = False
		if len(word_data_aligned_dic[key]) > 1:
			word_in_pieces = True

		for word_index, word_data in enumerate(word_data_aligned_dic[key]):
			wordEntry = {'sent.id':"", 'word.id':"", 'word.id.simple':"", 'word':"", 
					     'word.stripped':"", 'utt_pos':"", 'punc_before':"", 'punc_after':"", 
					     'minimal_punc_before': "", 'starttime':0, 'endtime':0, 'starttime.approx':0, 
					     'endtime.approx':0, 'features_f0':[0], 'pause_before_dur':0.0, 
					     'features_i0':[0], 'mean.f0_jump_from_prev':0.0, 'mean.i0_jump_from_prev':0.0,
					     'range.f0':0.0, 'range.i0':0.0}
			wordEntry['word.id.simple'] = key
			wordEntry['word.id'] = word_data[0]
			wordEntry['sent.id'] = word_data[1]
			print("%s,"%wordEntry['sent.id'])
			word_stripped = word_data[2]
			wordEntry['starttime'] = word_data[3]
			wordEntry['endtime'] = word_data[4]
			punc = word_data[5]

			if re.search(r"\w", word_stripped) == None:
				continue

			#strip word from non-word stuff 
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

			#skip speaker turn information
			if re.search(r"{|}", wordEntry['word']):
				continue

			#sometimes word file has the word transcription wrong. take care of those cases
			if re.search(r"\w", wordEntry['word']) == None:
				wordEntry['word'] = wordEntry['word.stripped']

			try:
				wordEntry['features_f0'] = word_id_to_f0_features_dic[wordEntry['word.id']]
			except Exception as e:
				wordEntry['features_f0'] = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
			
			try:
				wordEntry['features_i0'] = word_id_to_i0_features_dic[wordEntry['word.id']]
			except Exception as e:
				wordEntry['features_i0'] = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
			
			#silence values
			if not wordEntry['starttime'] == "NA":
				if not prev_wordEntry['endtime'] == "NA":
					diff = float(wordEntry['starttime']) - float(prev_wordEntry['endtime'])
					wordEntry['pause_before_dur'] = float("{0:.2f}".format(diff))

			#other prosodic features
			#jump.f0 = mean.f0 of the current word - mean.f0 of the previous word
			if not wordEntry['features_f0'][0] == "NA":
				if not prev_wordEntry['features_f0'][0] == "NA":
					wordEntry['mean.f0_jump_from_prev'] = float(wordEntry['features_f0'][0]) - float(prev_wordEntry['features_f0'][0])
			#jump.i0 = mean.i0 of the current word - mean.i0 of the previous word
			if not wordEntry['features_i0'][0] == "NA":
				if not prev_wordEntry['features_i0'][0] == "NA":
					wordEntry['mean.i0_jump_from_prev'] = float(wordEntry['features_i0'][0]) - float(prev_wordEntry['features_i0'][0])
			#range.f0 = max.f0 - min.f0
			if not wordEntry['features_f0'][2] == "NA" and not wordEntry['features_f0'][3] == "NA":
					wordEntry['range.f0'] = float(wordEntry['features_f0'][2]) - float(wordEntry['features_f0'][3])
			#range.i0 = max.i0 - min.i0
			if not wordEntry['features_i0'][2] == "NA" and not wordEntry['features_i0'][3] == "NA":
					wordEntry['range.i0'] = float(wordEntry['features_i0'][2]) - float(wordEntry['features_i0'][3])



			#convert i0 and f0 feature vectors to float vectors
			wordEntry['features_f0'] = featureVectorToFloat(wordEntry['features_f0'])
			wordEntry['features_i0'] = featureVectorToFloat(wordEntry['features_i0'])
			
			#check punctuation marks
			word_being_processed = wordEntry['word']
			punc_after = ""
			punc_before = ""

			#check beginning
			if re.search(r"^\W", word_being_processed) and word_index == 0:
				
				punc = word_being_processed[:re.search(r"\w", word_being_processed).start()]
					
				punc_before += punc
				#print(word_being_processed)
				word_being_processed = word_being_processed[re.search(r"\w", word_being_processed).start():]
				#print(wordEntry['word.stripped'])

			#check end again (issue with quotations)
			word_reversed = word_being_processed[::-1]
			if re.search(r"^\W",word_reversed) and word_index == len(word_data_aligned_dic[key]) - 1:
				punc = word_reversed[:re.search(r"\w", word_reversed).start()][::-1]
				punc_after = punc + punc_after
				word_being_processed = word_reversed[re.search(r"\w", word_reversed).start():][::-1]

			wordEntry['punc_before'] = punc_before
			wordEntry['punc_after'] = punc_after

			total_punc_before = prev_wordEntry['punc_after'] + wordEntry['punc_before']
			wordEntry['minimal_punc_before'] = puncProper(total_punc_before)

			#::::::::
			structured_data += [wordEntry]
			prev_wordEntry = wordEntry

	return structured_data

def dump_structured_word_data(structured_word_data, output_pickle_file):
	actualword_seq = []
	punc_before_seq = []
	pause_before_seq = []
	meanf0_seq = []
	medf0_seq = []
	meani0_seq = []
	slopef0_seq = []
	sdf0_seq = []
	jumpf0_seq = []
	jumpi0_seq = []
	rangef0_seq = []
	rangei0_seq = []

	for word_datum in structured_word_data:
		actualword_seq += [word_datum['word.stripped']]
		punc_before_seq += [word_datum['minimal_punc_before']]
		pause_before_seq += [word_datum['pause_before_dur']]
		meanf0_seq += [word_datum['features_f0'][0]]
		sdf0_seq += [word_datum['features_f0'][1]]
		medf0_seq += [word_datum['features_f0'][4]]
		slopef0_seq += [word_datum['features_f0'][14]]
		meani0_seq += [word_datum['features_i0'][0]]
		jumpf0_seq += [word_datum['mean.f0_jump_from_prev']]
		jumpi0_seq += [word_datum['mean.i0_jump_from_prev']]
		rangef0_seq += [word_datum['range.f0']]
		rangei0_seq += [word_datum['range.i0']]


	prosodic_punk_data = { 'word': actualword_seq,
						   'punctuation': punc_before_seq,
						   'pause': pause_before_seq,
						   'mean.f0': meanf0_seq,
						   'mean.i0': meani0_seq,
						   'med.f0': medf0_seq,
						   'slope.f0': slopef0_seq,
						   'sd.f0': sdf0_seq,
						   'jump.f0': jumpf0_seq,
						   'jump.i0': jumpi0_seq,
						   'range.f0': rangef0_seq,
						   'range.i0': rangei0_seq
	}

	with open(output_pickle_file, 'wb') as f:
		cPickle.dump(prosodic_punk_data, f, cPickle.HIGHEST_PROTOCOL)

def main(options):
	#checkFile(options.file_audio, "file_audio")
	checkFile(options.file_word, "file_word")
	checkFile(options.file_wordalign, "file_wordalign")
	checkFile(options.file_wordaggs_f0, "file_wordaggs_f0")
	checkFile(options.file_wordaggs_i0, "file_wordaggs_i0")

	[word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic] = readTedDataToMemory(options.file_word, options.file_wordalign, options.file_wordaggs_f0, options.file_wordaggs_i0)
	structured_word_data = structureData(word_data_simple_dic, word_id_to_f0_features_dic, word_id_to_i0_features_dic, word_data_aligned_dic)

	dump_structured_word_data(structured_word_data, options.file_output)

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-a", "--audio", dest="file_audio", default=None, help="wav", type="string")
	parser.add_option("-w", "--word", dest="file_word", default=None, help="word.txt", type="string")	#in /txt-sent
	parser.add_option("-l", "--align", dest="file_wordalign", default=None, help="word.txt.norm.align", type="string")	#in /txt-sent
	parser.add_option("-f", "--aggs_f0", dest="file_wordaggs_f0", default=None, help="aggs.alignword.txt under /derived/segs/f0/", type="string")	#in /derived/segs/f0/
	parser.add_option("-i", "--aggs_i0", dest="file_wordaggs_i0", default=None, help="aggs.alignword.txt under /derived/segs/i0/", type="string")	#in /derived/segs/i0/
	#parser.add_option("-p", "--pause", dest="write_pauses", default=False, help="prints pauses between words", action="store_true")
	parser.add_option("-o", "--out", dest="file_output", default=None, help="outputfile", type="string")
	#parser.add_option("-p", "--out_pause", dest="file_output_pause", default=None, help="outputfile_pause", type="string")

	(options, args) = parser.parse_args()
	main(options)
	#print("Done.")