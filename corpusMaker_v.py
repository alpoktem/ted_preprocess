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
import tedDataToPickle

reload(sys)  
sys.setdefaultencoding('utf8')

#PUNCTUATION_VOCABULARY_EXTENDED = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH", '"QUOTE', 
#	':"COLONQUOTE', ',"COMMAQUOTE', '!"EXCLAMATIONQUOTE', '."PERIODQUOTE', '?"QUESTIONQUOTE', '":QUOTECOLON', 
#	'",QUOTECOMMA', '"!QUOTEEXCLAMATION', '"?QUOTEQUESTION', '""QUOTEQUOTE', '";QUOTESEMICOLON', ';"SEMICOLONQUOTE']

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"

MAX_WORD_VOCABULARY_SIZE = 100000

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

def dump_data_to_pickle(data, output_pickle_file):
	with open(output_pickle_file, 'wb') as f:
		cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def read_vocabulary(file_name):
	with codecs.open(file_name, 'r', 'utf-8') as f:
		return iterable_to_dict(f.readlines())

def get_word_counts_from_samples(samples):
	word_counts = dict()
	for sample in samples:
		for word in sample['word']:
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

def initialize_sample(sample_sequence_length):
	sample = {'word.id':[-1] * sample_sequence_length,
			  'word': [''] * (sample_sequence_length - 1) + [END],
			  'word.duration': [0.0] * sample_sequence_length,
			  'speech.rate.norm': [0.0] * sample_sequence_length,
			  'punctuation': [''] * sample_sequence_length,
			  'punctuation.reduced': [''] * sample_sequence_length,
			  'punc.id': [0] * sample_sequence_length,
			  'punc.red.id': [0] * sample_sequence_length,
			  'pause.id': [0] * sample_sequence_length,
			  'mean.f0.id': [0] * sample_sequence_length,
			  'mean.i0.id': [0] * sample_sequence_length,
			  'range.f0.id': [0] * sample_sequence_length,
			  'range.i0.id': [0] * sample_sequence_length,
			  'mean.f0':[0.0] * sample_sequence_length,
			  'mean.i0':[0.0] * sample_sequence_length,
			  'range.f0':[0.0] * sample_sequence_length,
			  'range.i0':[0.0] * sample_sequence_length}
	return sample

def initialize_test_data():
	test_data = {
			  'word': [],
			  'word.duration': [],
			  'speech.rate.norm': [],
			  'punctuation': [],
			  'punctuation.reduced': [],
			  'punc.id': [],
			  'punc.red.id': [],
			  'pause.id': [],
			  'mean.f0.id': [],
			  'mean.i0.id': [],
			  'range.f0.id': [],
			  'range.i0.id': [],
			  'mean.f0':[],
			  'mean.i0':[],
			  'range.f0':[],
			  'range.i0':[] }
	return test_data

def sample_data_from_files(talkfiles, sample_sequence_length, desired_no_of_samples):
	samples = []
	desired_no_of_samples_reached = False
	extract_data = []
	for talkfile in talkfiles:
		if desired_no_of_samples_reached:
			break
		sample_id_start = len(samples)
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)
			talkdata_size = len(talkdata['word'])
			talkdata_seq_idx_start = 0
			talkdata_seq_idx_end = sample_sequence_length - 1

			while talkdata_seq_idx_end < talkdata_size and not desired_no_of_samples_reached:
				current_sample = initialize_sample(sample_sequence_length)
				sample_index = 0
				last_eos_idx = 0
				for talkdata_seq_index in range(talkdata_seq_idx_start, talkdata_seq_idx_end):
					curr_punc_code = talkdata['punc.id'][talkdata_seq_index]
					current_sample['punc.id'][sample_index] = curr_punc_code
					current_sample['punc.red.id'][sample_index] = talkdata['punc.red.id'][talkdata_seq_index]

					if curr_punc_code in tedDataToPickle.EOS_PUNCTUATION_CODES:
						last_eos_idx = talkdata_seq_index
						#print("last_eos:%s"%last_eos_idx)
					
					for key in current_sample.keys():
						if key == 'word.id':
							continue
						current_sample[key][sample_index] = talkdata[key][talkdata_seq_index]
					
					sample_index += 1
				if sample_index == sample_sequence_length - 1:
					if talkdata_seq_index < talkdata_size:
						current_sample['punctuation'][sample_index] = talkdata['punctuation'][talkdata_seq_index+1]
						current_sample['punc.id'][sample_index] = talkdata['punc.id'][talkdata_seq_index+1]
						current_sample['punc.red.id'][sample_index] = talkdata['punc.red.id'][talkdata_seq_index+1]

					samples.append(current_sample)
					
					if len(samples) >= desired_no_of_samples:
						desired_no_of_samples_reached = True

				if last_eos_idx > talkdata_seq_idx_start:
					talkdata_seq_idx_start = last_eos_idx
					talkdata_seq_idx_end = talkdata_seq_idx_start + sample_sequence_length - 1
				else:
					eos_search_idx = talkdata_seq_idx_end
					eos_found_before_end = False
					while eos_search_idx < talkdata_size:
						if talkdata['punc.id'][eos_search_idx] in tedDataToPickle.EOS_PUNCTUATION_CODES:
							last_eos_idx = eos_search_idx
							talkdata_seq_idx_start = last_eos_idx
							talkdata_seq_idx_end = talkdata_seq_idx_start + sample_sequence_length - 1
							eos_found_before_end = True
							break
						eos_search_idx += 1
					if not eos_found_before_end:
						talkdata_seq_idx_end = talkdata_size
		sample_id_end = len(samples) - 1
		extract_info = [talkfile, sample_id_start, sample_id_end]
		print(extract_info)
		extract_data.append(extract_info)
	return [samples, extract_data]

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
	#test_set = all_samples[dev_end: no_of_samples]
	take_next = False
	test_set_files = []

	for extract_info in extract_data:
		if extract_info[1] > dev_end:
			take_next = True
		if take_next:
			test_set_files.append(extract_info[0])

	return [training_set, dev_set, test_set_files]

def extract_uncut_test(test_set_files):
	test_data = initialize_test_data()
	for talkfile in test_set_files:
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)

			for key in test_data.keys():
				if key == 'word.id':
					continue
				test_data[key].extend(talkdata[key])
	return [test_data]


def add_word_ids_to_samples(samples, word_vocabulary):
	for sample in samples:
		sequence_length = len(sample['word'])
		sample['word.id'] = [0]*sequence_length
		for index, word in enumerate(sample['word']):
			sample['word.id'][index] = word_vocabulary.get(word, word_vocabulary[UNK])

def convert_value_to_level(pause_dur, pause_bins):
	level = 0
	for bin_no, bin_upper_limit in enumerate(pause_bins):
		if pause_dur > bin_upper_limit:
			level += 1
		else:
			break
	return level

def main(options):
	if not checkArgument(options.input_dir, isDir=True):
		sys.exit("Input directory missing")
	if not checkArgument(options.output_dir, isDir=True, createDir=True):
		sys.exit("Output directory missing")
	else:
		ALL_FILE_PICKLE = os.path.join(options.output_dir, "all.pickle")
		ALL_FILE_CSV = os.path.join(options.output_dir, "all.csv")
		TRAIN_FILE_PICKLE = os.path.join(options.output_dir, "train.pickle")
		TRAIN_FILE_CSV = os.path.join(options.output_dir, "train.csv")
		TEST_FILE_PICKLE = os.path.join(options.output_dir, "test.pickle")
		TEST_FILE_CSV = os.path.join(options.output_dir, "test.csv")
		DEV_FILE_PICKLE = os.path.join(options.output_dir, "dev.pickle")
		DEV_FILE_CSV = os.path.join(options.output_dir, "dev.csv")
		WORD_VOCAB_FILE = os.path.join(options.output_dir, "vocabulary.pickle")
		METADATA_FILE_PICKLE = os.path.join(options.output_dir, "metadata.pickle")
		METADATA_FILE_CSV = os.path.join(options.output_dir, "metadata.csv")
	if checkArgument(options.set_size):
		max_set_size = options.set_size
	else:
		max_set_size = 9999999999999

	talkfiles = glob.glob(options.input_dir + '/*.pickle')
	talkfiles.sort()

	#merge all data in files while segmenting to samples of size sequence_length. Each sample should start with a new sentence. Stop when set_size is reached 
	[all_samples, extraction_data] = sample_data_from_files(talkfiles, options.sequence_length, max_set_size)

	word_counts = get_word_counts_from_samples(all_samples)
	vocabulary = build_vocabulary(word_counts, options.min_vocab)

	write_vocabulary(vocabulary, WORD_VOCAB_FILE)
	word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
	print("Vocabulary extracted to %s. (Size: %i)"%(WORD_VOCAB_FILE, len(word_vocabulary)))

	add_word_ids_to_samples(all_samples, word_vocabulary)
	
	[training_set, dev_set, test_set_files] = split_data_to_sets(all_samples, options.train_split_ratio, extraction_data)
	testing_set = extract_uncut_test(test_set_files)
	add_word_ids_to_samples(testing_set, word_vocabulary)  #test set is like one sample

	#dump_data_to_pickle(all_set, ALL_FILE_PICKLE)
	#print("All samples dumped to %s (Size: %i)"%(ALL_FILE_PICKLE, len(all_set)))
	dump_data_to_pickle(training_set, TRAIN_FILE_PICKLE)
	print("Training samples dumped to %s (Size: %i)"%(TRAIN_FILE_PICKLE, len(training_set)))
	dump_data_to_pickle(dev_set, DEV_FILE_PICKLE)
	print("Development samples dumped to %s (Size: %i)"%(DEV_FILE_PICKLE, len(dev_set)))
	dump_data_to_pickle(testing_set, TEST_FILE_PICKLE)
	print("Unsampled testing data dumped to %s (Size: %i)"%(TEST_FILE_PICKLE, len(testing_set)))

	#prepare corpus metadata
	with open(talkfiles[0], 'rb') as f:
		talkdata = cPickle.load(f)
		talkdata_metadata = talkdata['metadata']
	metadata = {'word_vocab_size': len(word_vocabulary),
				'punc_vocab_size': len(tedDataToPickle.PUNCTUATION_VOCABULARY),
				'punc_red_vocab_size': len(tedDataToPickle.REDUCED_PUNCTUATION_VOCABULARY),
				'min_occurence_for_vocab': options.min_vocab,
				#'all_file_path': ALL_FILE_PICKLE,
				'train_file_path': TRAIN_FILE_PICKLE,
				'test_file_path': TEST_FILE_PICKLE,
				'dev_file_path': DEV_FILE_PICKLE,
				#'all_set_size': len(all_set),
				'training_set_size': len(training_set),
				'dev_set_size': len(dev_set),
				'test_set_size': len(testing_set),
				'test_set_files': test_set_files,
				'sequence_length': options.sequence_length,
				'no_of_semitone_levels': talkdata_metadata['no_of_semitone_levels'],
				'no_of_pause_levels': talkdata_metadata['no_of_pause_levels']
	}
	dump_data_to_pickle(metadata, METADATA_FILE_PICKLE)
	print("Corpus metadata dumped to %s"%(METADATA_FILE_PICKLE))

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-i", "--input_dir", dest="input_dir", default=None, help="Input directory with a pickle file for each talk", type="string")
	parser.add_option("-o", "--output_dir", dest="output_dir", default=None, help="Output directory to put train, dev, test files")
	parser.add_option("-r", "--train_ratio", dest="train_split_ratio", default=0.7, help="Ratio of samples to put for training (default=0.7)", type="float")
	parser.add_option("-s", "--set_size", dest="set_size", help="number of samples to process", type="int")
	parser.add_option("-v", "--min_vocab", dest="min_vocab", default=3, help="min number of word occurence to be added into vocabulary", type="int")
	parser.add_option("-t", "--write_test_text", dest="write_test_text", default=False, help="write text data for scoring", action="store_true")
	parser.add_option("-l", "--sequence_length", dest="sequence_length", default=50, help="sequence length", type="int")

	(options, args) = parser.parse_args()
	main(options)