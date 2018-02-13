#merges all pickle files in a directory, extracts vocabulary, segments into samples of constant size and splits into training, testing and development sets 
# -*- coding: utf-8 -*-
from optparse import OptionParser
import os
import sys
import cPickle
import operator
import codecs
import glob
import numpy as np
import tedDataToCsv
import csv
from collections import defaultdict
import nltk
import shutil

reload(sys)  
sys.setdefaultencoding('utf8')

#PUNCTUATION_VOCABULARY_EXTENDED = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH", '"QUOTE', 
#	':"COLONQUOTE', ',"COMMAQUOTE', '!"EXCLAMATIONQUOTE', '."PERIODQUOTE', '?"QUESTIONQUOTE', '":QUOTECOLON', 
#	'",QUOTECOMMA', '"!QUOTEEXCLAMATION', '"?QUOTEQUESTION', '""QUOTEQUOTE', '";QUOTESEMICOLON', ';"SEMICOLONQUOTE']

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"

SPACE="_"
PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?', 4:'!', 5:'-', 6:';', 7:':'}

MAX_WORD_VOCABULARY_SIZE = 100000

CSV_COLUMNS = ['word', 'punctuation_before', 'pos', 'pause_before', 'f0_mean', 'f0_range', 'f0_birange', 'f0_sd',  'i0_mean', 'i0_range', 'i0_birange',  'i0_sd', 'speech_rate_norm']

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def checkArgument(argname, isFile=False, isDir=False, createDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir:
			if not os.path.isdir(argname):
				if createDir:
					print("Creating directory %s"%(argname))
					os.makedirs(argname)
				else:
					return False
	return True

def build_vocabulary(word_counts, min_word_count):
	return [wc[0] for wc in reversed(sorted(word_counts.items(), key=operator.itemgetter(1))) if wc[1] >= min_word_count and wc[0] != UNK][:MAX_WORD_VOCABULARY_SIZE] # Unk will be appended to end

def iterable_to_dict(arr):
	return dict((x.strip(), i) for (i, x) in enumerate(arr))

def read_vocabulary(file_name):
	with codecs.open(file_name, 'r', 'utf-8') as f:
		return iterable_to_dict(f.readlines())

def get_word_counts_from_samples(samples):
	word_counts = dict()
	for sample in samples:
		for word in sample['word']:
			if not word == END:
				word_counts[word] = word_counts.get(word, 0) + 1
	return word_counts

def write_vocabulary(vocabulary, file_name):
	if END not in vocabulary:
		vocabulary.append(END)
	if UNK not in vocabulary:
		vocabulary.append(UNK)
	if EMP not in vocabulary:
		vocabulary.append(EMP)

	with codecs.open(file_name, 'w', 'utf-8') as f:
		f.write("\n".join(vocabulary))

def get_word_counts(file_list):
	word_counts = dict()
	for talkfile in file_list:
		#print(talkfile)
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)
			for word in talkdata['word']:
				word_counts[word] = word_counts.get(word, 0) + 1
	return word_counts

def initialize_empty_sample():

	sample = {key:[] for key in CSV_COLUMNS }
	return sample

def read_proscript_as_dict(filename):
	dict = {}
	columns = defaultdict(list) # each value in each column is appended to a list

	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='\t') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			for (k,v) in row.items(): # go over each column name and value 
				if k == "word" or "punctuation" in k or "pos" in k:
					columns[k].append(v) # append the value into the appropriate list
				else:
					try:
						columns[k].append(float(v)) # real value
					except ValueError:
						print("ALARM:%s"%v)
						columns[k].append(0.0)
	return columns

def read_proscript_as_list(filename):
	proscript = []

	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='|') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			proscript.append({k:v for (k,v) in row.items()})

	return proscript

def sample_variable_length_data_from_files(talkfiles, max_sample_length, no_of_talks_to_process):
	samples = []
	extract_data = []
	for talkfile in talkfiles:
		if len(extract_data) == no_of_talks_to_process:
			break

		sample_id_start = len(samples)
		talk_proscript = read_proscript_as_list(talkfile)

		talkdata_size = len(talk_proscript)
		talkdata_seq_idx = 0
		last_eos_idx = 0
		last_eos = ""
		curr_sentence = []
		curr_sample = initialize_empty_sample()

		while talkdata_seq_idx < talkdata_size:
			if talk_proscript[talkdata_seq_idx]['punctuation_before'] in tedDataToCsv.EOS_PUNCTUATION and not len(curr_sentence) >= max_sample_length - 1:
				#print("eos_punctuation_before")
				#if sentence fits
				#print(last_eos_idx)
				#raw_input("Press")
				#if the sentence fits
				if len(curr_sample['word']) + len(curr_sentence) < max_sample_length - 1:
					#add sentence to sample
					#print("sentence fits add to sample")
					add_sentence_to_sample(curr_sample, curr_sentence)
					#raw_input("Press")
					curr_sentence = []
					curr_sentence.append(talk_proscript[talkdata_seq_idx])
					last_eos_idx = talkdata_seq_idx
					last_eos = talk_proscript[talkdata_seq_idx]['punctuation_before']
					talkdata_seq_idx += 1
					#print(last_eos_idx)
				elif len(curr_sample) > 0: 
					#print("sample full (%i). add to samples. continue search from %s"%(len(curr_sample['word']),talkdata_seq_idx))
					finish_sample(curr_sample, last_eos)
					samples.append(curr_sample)
					#print("no_of_samples: %i"%len(samples))
					curr_sample = initialize_empty_sample()	
			elif len(curr_sentence) >= max_sample_length - 1:
				#print(talkdata_seq_idx)
				while not talk_proscript[talkdata_seq_idx]['punctuation_before'] in tedDataToCsv.EOS_PUNCTUATION:
					talkdata_seq_idx += 1
					if talkdata_seq_idx >= talkdata_size:
						break
				curr_sentence = []
			else:
				#fill current sentence
				curr_sentence.append(talk_proscript[talkdata_seq_idx])
				talkdata_seq_idx += 1

		sample_id_end = len(samples) - 1
		extract_info = [talkfile, sample_id_start, sample_id_end]
		print(extract_info)
		extract_data.append(extract_info)
	return [samples, extract_data]

def finish_sample(sample, last_punc):
	for key in sample:
		if key == 'word':
			sample[key].append(END)
		elif 'punctuation' in key:
			sample[key].append(last_punc)
		elif 'pos' in key:
			sample[key].append(END)
		else:
			sample[key].append(0.0)

def add_sentence_to_sample(sample, sentence):
	sentence_tokens = []
	for word_data in sentence:
		sentence_tokens.append(word_data['word']) 

	pos_data = nltk.pos_tag(sentence_tokens)

	for idx, word_data in enumerate(sentence):
		for key in word_data:
			sample[key].append(word_data[key])
		sample['pos'].append(pos_data[idx][1])

#Splits test set as well
def split_data_to_sets(all_samples, train_split_ratio, extract_data):
	no_of_samples = len(all_samples)
	
	training_size = int(no_of_samples * train_split_ratio)
	testing_size = int((no_of_samples - training_size) / 2)
	dev_size = no_of_samples - training_size - testing_size

	all_set = all_samples[0:no_of_samples]
	training_set = all_samples[0:training_size]
	dev_start = training_size + 1
	dev_end = training_size + dev_size + 1
	dev_set = all_samples[dev_start:dev_end]
	test_set = all_samples[dev_end: no_of_samples]

	return [training_set, dev_set, test_set]

def dump_data_to_csv(data, output_csv_file):
	with open(output_csv_file, 'wb') as f:
		w = csv.writer(f, delimiter="|")
		rowIds = CSV_COLUMNS
		w.writerow(rowIds)
		for index in range(len(data['word'])):  
			try:
				row = [data[key][index] for key in CSV_COLUMNS]
				w.writerow(row) 
			except:
				print("Problem with %s"%output_csv_file)

def dump_samples_to_dir(samples, output_directory, as_proscript=False, as_groundtruth=False):
	if not os.path.isdir(output_directory):
		print("Creating directory %s"%(output_directory))
		os.makedirs(output_directory)
	for sample_no, sample in enumerate(samples):
		if as_proscript:
			sample_file = os.path.join(output_directory, "%i.csv"%sample_no)
			dump_data_to_csv(sample, sample_file)
		elif as_groundtruth:
			sample_file = os.path.join(output_directory, "%i.txt"%sample_no)
			dump_data_to_text(sample, sample_file)

def dump_data_to_text(data, output_file):
	with codecs.open(output_file, 'w', 'utf-8') as f_out:
		for index, word in enumerate(data['word']):
			if not index == 0:
				if data['punctuation_before'][index]:
					f_out.write(data['punctuation_before'][index] + " ")
			if not word == END:
				f_out.write(word + " ")

def main(options):
	if not checkArgument(options.input_dir, isDir=True):
		sys.exit("Input directory missing")
	if not checkArgument(options.output_dir, isDir=True, createDir=True):
		sys.exit("Output directory missing")
	else:
		TRAIN_FILE_CSV_DIR = os.path.join(options.output_dir, "train_samples")
		TRAIN_FILE_ARCHIVE = os.path.join(options.output_dir, "train_samples")
		TEST_FILE_CSV_DIR = os.path.join(options.output_dir, "test_samples")
		TEST_FILE_ARCHIVE = os.path.join(options.output_dir, "test_samples")
		TEST_TXT_DIR = os.path.join(options.output_dir, "test_groundtruth")
		TEST_TXT_ARCHIVE = os.path.join(options.output_dir, "test_groundtruth")
		DEV_FILE_CSV_DIR = os.path.join(options.output_dir, "dev_samples")
		DEV_FILE_ARCHIVE = os.path.join(options.output_dir, "dev_samples")
		WORD_VOCAB_FILE = os.path.join(options.output_dir, "vocabulary.txt")
	
	if checkArgument(options.set_size):
		max_set_size = options.set_size
	else:
		max_set_size = 9999999999999

	talkfiles_csv = glob.glob(options.input_dir + '/*.csv')
	talkfiles_csv.sort()

	[all_samples, extraction_data] = sample_variable_length_data_from_files(talkfiles_csv, options.max_sequence_length, max_set_size)

	word_counts = get_word_counts_from_samples(all_samples)
	vocabulary = build_vocabulary(word_counts, options.min_vocab)

	write_vocabulary(vocabulary, WORD_VOCAB_FILE)
	word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
	print("Vocabulary extracted to %s. (Size: %i)"%(WORD_VOCAB_FILE, len(word_vocabulary)))

	print("Total number of samples: %i"%len(all_samples))
	[training_set, dev_set, test_set] = split_data_to_sets(all_samples, options.train_split_ratio, extraction_data)

	dump_samples_to_dir(training_set, TRAIN_FILE_CSV_DIR, as_proscript=True)
	if options.archive_samples:
		shutil.make_archive(TRAIN_FILE_ARCHIVE, format='tar', root_dir=options.output_dir, base_dir="train_samples")
		shutil.rmtree(TRAIN_FILE_CSV_DIR)
		print("Training samples dumped to %s (Size: %i)"%(TRAIN_FILE_ARCHIVE, len(training_set)))
	else:
		print("Training samples dumped to %s (Size: %i)"%(TRAIN_FILE_CSV_DIR, len(training_set)))

	dump_samples_to_dir(dev_set, DEV_FILE_CSV_DIR, as_proscript=True)
	if options.archive_samples:
		shutil.make_archive(DEV_FILE_ARCHIVE, format='tar', root_dir=options.output_dir, base_dir="dev_samples")
		shutil.rmtree(DEV_FILE_CSV_DIR)
		print("Development samples dumped to %s (Size: %i)"%(DEV_FILE_ARCHIVE, len(dev_set)))
	else:
		print("Development samples dumped to %s (Size: %i)"%(DEV_FILE_CSV_DIR, len(dev_set)))

	dump_samples_to_dir(test_set, TEST_FILE_CSV_DIR, as_proscript=True)
	if options.archive_samples:
		shutil.make_archive(TEST_FILE_ARCHIVE, format='tar', root_dir=options.output_dir, base_dir="test_samples")
		shutil.rmtree(TEST_FILE_CSV_DIR)
		print("Testing samples dumped to %s (Size: %i)"%(TEST_FILE_ARCHIVE, len(test_set)))
	else:
		print("Testing samples dumped to %s (Size: %i)"%(TEST_FILE_CSV_DIR, len(test_set)))

	dump_samples_to_dir(test_set, TEST_TXT_DIR, as_groundtruth=True)
	if options.archive_samples:
		shutil.make_archive(TEST_TXT_ARCHIVE, format='tar', root_dir=options.output_dir, base_dir="test_groundtruth")
		shutil.rmtree(TEST_TXT_DIR)
		print("Testing samples dumped to %s (Size: %i)"%(TEST_TXT_ARCHIVE, len(test_set)))
	else:
		print("Testing samples dumped to %s (Size: %i)"%(TEST_TXT_DIR, len(test_set)))

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-i", "--input_dir", dest="input_dir", default=None, help="Input directory with proscripts of each talk", type="string")
	parser.add_option("-o", "--output_dir", dest="output_dir", default=None, help="Output directory to put train, dev, test files")
	parser.add_option("-r", "--train_ratio", dest="train_split_ratio", default=0.7, help="Ratio of samples to put for training (default=0.7)", type="float")
	parser.add_option("-s", "--set_size", dest="set_size", help="number of talks to process (by default all)", type="int")
	parser.add_option("-v", "--min_vocab", dest="min_vocab", default=3, help="min number of word occurence to be added into vocabulary", type="int")
	parser.add_option("-l", "--max_sequence_length", dest="max_sequence_length", default=50, help="sequence length", type="int")
	parser.add_option("-z", "--archive_samples", dest="archive_samples", default=False, help="if sample directories should be archived", action="store_true")

	(options, args) = parser.parse_args()
	main(options)