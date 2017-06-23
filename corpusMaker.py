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

reload(sys)  
sys.setdefaultencoding('utf8')

#PUNCTUATION_VOCABULARY_EXTENDED = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH", '"QUOTE', 
#	':"COLONQUOTE', ',"COMMAQUOTE', '!"EXCLAMATIONQUOTE', '."PERIODQUOTE', '?"QUESTIONQUOTE', '":QUOTECOLON', 
#	'",QUOTECOMMA', '"!QUOTEEXCLAMATION', '"?QUOTEQUESTION', '""QUOTEQUOTE', '";QUOTESEMICOLON', ';"SEMICOLONQUOTE']

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"
SPACE = "_"

PUNCTUATION_CODES =  {0:'?', 1:'!', 2:SPACE, 3:',', 4:'-', 5:':', 6:';', 7:'.' }
INV_PUNCTUATION_CODES = {'?':0, '!':1, SPACE:2, ',':3, '-':4, ':':5, ';':6, '.':7, '':2}

EOS_PUNCTUATION_CODES = [0,1,7]
EOS_PUNCTUATION = ['?', '!', '.']

REDUCED_PUNCTUATION_CODES =  {0:SPACE, 1:'.', 2:',', 3:'?'}
REDUCED_INV_PUNCTUATION_CODES = {SPACE:0, '.':1, ',':2, '?':3}

MAX_WORD_VOCABULARY_SIZE = 100000
MAX_SEQUENCE_LEN = 50

JUMPF0_PREFIX = "<jump.f0="
JUMPI0_PREFIX = "<jump.i0="
RANGEF0_PREFIX = "<range.f0="
RANGEI0_PREFIX = "<range.i0="
SLOPEF0_PREFIX = "<slope.f0="
PAUSE_PREFIX = "<sil="
MEANF0_PREFIX = "<mean.f0="
MEANI0_PREFIX = "<mean.i0="

PARAM_SUFFIX = ">"

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

def write_vocabulary(vocabulary, file_name):
	if END not in vocabulary:
		vocabulary.append(END)
	if UNK not in vocabulary:
		vocabulary.append(UNK)
	if EMP not in vocabulary:
		vocabulary.append(EMP)

	with codecs.open(file_name, 'w', 'utf-8') as f:
		f.write("\n".join(vocabulary))


def reducePuncCode(puncCode):
	if puncCode == 2:  #space
		return 0
	if puncCode in [1, 4, 5, 6, 7]: #period
		return 1
	elif puncCode == 3: #comma
		return 2
	elif puncCode == 0: #question
		return 3

def reducePuncInTalkdata(punc):
	if punc == '':
		return '_'
	elif punc in ['!', '-', ':', ';', '.']:
		return '.'
	else:
		return punc

def initialize_sample(original_data):
	current_sample = dict.fromkeys(original_data)
	for key in current_sample.keys():
		current_sample[key] = []
	current_sample['word.id'] = []
	current_sample['punc.id'] = []
	current_sample['punc.red.id'] = []
	current_sample['pause.id'] = []
	current_sample['mean.f0.id'] = []
	current_sample['mean.i0.id'] = []
	current_sample['jump.f0.id'] = []
	current_sample['jump.i0.id'] = []
	current_sample['range.f0.id'] = []
	current_sample['range.i0.id'] = []
	current_sample['slope.f0.id'] = []

	return current_sample
	
def endSample(sample, next_punctuation = ''):
	for key in sample.keys():
		if key == 'word.id':
			sample[key].append(-1)
		elif key == 'word':
			sample[key].append(END)
		elif key == 'punctuation':
			sample[key].append(next_punctuation)
		elif key == 'punc.id':
			sample[key].append(INV_PUNCTUATION_CODES[next_punctuation])
		elif key == 'punc.red.id':
			sample[key].append(reducePuncCode(INV_PUNCTUATION_CODES[next_punctuation]))
		elif key == 'pause.id':
			pass
		elif key == 'mean.f0.id':
			pass
		elif key == 'mean.i0.id':
			pass
		elif key == 'jump.f0.id':
			pass
		elif key == 'jump.i0.id':
			pass
		elif key == 'slope.f0.id':
			pass
		elif key == 'range.f0.id':
			pass
		elif key == 'range.i0.id':
			pass
		else:
			sample[key].append(0.0)

def get_word_counts(file_list):
	word_counts = dict()
	for talkfile in file_list:
		#print(talkfile)
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)
			for word in talkdata['word']:
				word_counts[word] = word_counts.get(word, 0) + 1
	return word_counts

def get_word_counts_from_samples(samples):
	word_counts = dict()
	for sample in samples:
		for word in sample['word']:
			word_counts[word] = word_counts.get(word, 0) + 1
	return word_counts

def write_test_text_data(test_text_file, test_text_plain_file, extraction_data, test_sample_start, test_sample_end):
	#allocate data in talkfiles corresponding between test_sample_start and test_sample_end as testing data. this will be written as text file as punctuator.py works like that now.
	talks_for_test = []
	fortest = False
	for index, data in enumerate(extraction_data):
		if fortest or data[1] == test_sample_start:
			talks_for_test.append(data[0])
			fortest = True

		if data[1] < test_sample_start and data[2] > test_sample_start:
			fortest = True

		if data[1] <= test_sample_end and data[2] > test_sample_end:
			talks_for_test.append(data[0])
			break

	wordcount = 0	
	for talkfile in talks_for_test:
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)
			talkdata_size = len(talkdata['word'])
			for idx in xrange(talkdata_size):
				punctuation = talkdata['punctuation'][idx]
				#reduced_punctuation_code = talkdata['punc.red.id'][idx]
				
				word_write_plain = ""
				word_write_params = "%s%.3f%s %s%.3f%s %s%.3f%s %s%.3f%s %s%.3f%s "%(MEANF0_PREFIX, talkdata['mean.f0'][idx], PARAM_SUFFIX, 
															MEANI0_PREFIX, talkdata['mean.i0'][idx], PARAM_SUFFIX,
															RANGEF0_PREFIX, talkdata['range.f0'][idx], PARAM_SUFFIX,
															RANGEI0_PREFIX, talkdata['range.i0'][idx], PARAM_SUFFIX,
															PAUSE_PREFIX, talkdata['pause'][idx], PARAM_SUFFIX
															)
				# if punctuation:
				# 	word_write_params += "%s "%(punctuation)
				# 	word_write_plain += "%s "%(punctuation)
				# else:
				# 	word_write_params += "%s "%(SPACE)
				# 	word_write_plain += "%s "%(SPACE)

				word_write_params += "%s "%(reducePuncInTalkdata(punctuation))
				word_write_plain += "%s "%(reducePuncInTalkdata(punctuation))

				word_write_params += "%s "%(talkdata['word'][idx])
				word_write_plain += "%s "%(talkdata['word'][idx])

				test_text_file.write(word_write_params)
				test_text_plain_file.write(word_write_plain)
				wordcount += 1
	return [talks_for_test, wordcount]



def sample_data_from_files(talkfiles, desired_no_of_samples):
	samples = []
	desired_no_of_samples_reached = False
	extract_data = []
	for talkfile in talkfiles:
		if desired_no_of_samples_reached:
			break
		sample_id_start = len(samples)
		#print(talkfile)
		#print("samples start: %i"%len(samples))
		with open(talkfile, 'rb') as f:
			talkdata = cPickle.load(f)
			talkdata_size = len(talkdata['word'])
			talkdata_seq_idx_start = 0
			talkdata_seq_idx_end = MAX_SEQUENCE_LEN - 1

			while talkdata_seq_idx_end < talkdata_size and not desired_no_of_samples_reached:
				
				current_sample = initialize_sample(talkdata)
				current_sample_size = 0
				last_eos_idx = 0
				for seq_index in range(talkdata_seq_idx_start, talkdata_seq_idx_end):
					
					#for seq_index, actualword in enumerate(talkdata['word']):
					#word_id = word_vocabulary.get(talkdata['word'][seq_index], word_vocabulary[UNK])
					current_sample['word.id'].append(-1)
					curr_punc_code = INV_PUNCTUATION_CODES[talkdata['punctuation'][seq_index]]
					current_sample['punc.id'].append(curr_punc_code)
					current_sample['punc.red.id'].append(reducePuncCode(curr_punc_code))

					if curr_punc_code in EOS_PUNCTUATION_CODES:
						last_eos_idx = seq_index
						#print("last_eos:%s"%last_eos_idx)
					
					for key in talkdata.keys():
						current_sample[key].append(talkdata[key][seq_index])
					
					current_sample_size += 1
				if current_sample_size == MAX_SEQUENCE_LEN - 1:
					if seq_index < talkdata_size:
						next_punctuation = talkdata['punctuation'][seq_index+1]
					endSample(current_sample, next_punctuation)
					
					samples.append(current_sample)
					#print("sample no.%s added"%(len(samples)-1))
					#print("%i - %i"%(talkdata_seq_idx_start, talkdata_seq_idx_end))
					current_sample = initialize_sample(talkdata)
					current_sample_size = 0

					if len(samples) >= desired_no_of_samples:
						desired_no_of_samples_reached = True

				if last_eos_idx > talkdata_seq_idx_start:
					talkdata_seq_idx_start = last_eos_idx
					talkdata_seq_idx_end = talkdata_seq_idx_start + MAX_SEQUENCE_LEN - 1
				else:
					eos_search_idx = talkdata_seq_idx_end
					eos_found_before_end = False
					while eos_search_idx < talkdata_size:
						if talkdata['punctuation'][eos_search_idx] in EOS_PUNCTUATION:
							last_eos_idx = eos_search_idx
							talkdata_seq_idx_start = last_eos_idx
							talkdata_seq_idx_end = talkdata_seq_idx_start + MAX_SEQUENCE_LEN - 1
							eos_found_before_end = True
							break
						eos_search_idx += 1
					if not eos_found_before_end:
						talkdata_seq_idx_end = talkdata_size
		#print("samples end: %i"%len(samples))
		sample_id_end = len(samples) - 1
		extract_info = [talkfile, sample_id_start, sample_id_end]
		print(extract_info)
		extract_data.append(extract_info)
	return [samples, extract_data]

def split_data_to_sets(all_samples, train_split_ratio):
	no_of_samples = len(all_samples)
	
	training_size = int(no_of_samples * train_split_ratio)
	testing_size = int((no_of_samples - training_size) / 2)
	dev_size = no_of_samples - training_size - testing_size

	all_set = all_samples[0:no_of_samples]
	training_set = all_samples[0:training_size]
	testing_start = training_size + 1
	testing_end = training_size + testing_size + 1
	testing_set = all_samples[testing_start:testing_end]
	dev_set = all_samples[testing_end: no_of_samples]

	return [all_set, training_set, testing_set, dev_set, testing_start, testing_end]

def add_word_ids_to_samples(samples, word_vocabulary):
	for sample in samples:
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

def add_levels_to_samples(samples, logfile):
	pause_bins = create_pause_bins()
	print("Number of pause levels: %i. "%len(pause_bins))
	logfile.write("Number of pause levels: %i. \n"%len(pause_bins))
	semitone_bins = create_semitone_bins()
	print("Number of semitone levels: %i. "%len(semitone_bins))
	logfile.write("Number of semitone levels: %i. \n"%len(semitone_bins))
	slope_bins = create_slope_bins()
	print("Number of slope levels: %i. "%len(slope_bins))
	logfile.write("Number of semitone levels: %i. \n"%len(semitone_bins))

	for sample in samples:
		for index, pause_dur in enumerate(sample['pause']):
			pause_level = convert_value_to_level(pause_dur, pause_bins)
			sample['pause.id'].append(pause_level)

			meanF0_level = convert_value_to_level(sample['mean.f0'][index], semitone_bins)
			sample['mean.f0.id'].append(meanF0_level)

			meanI0_level = convert_value_to_level(sample['mean.i0'][index], semitone_bins)
			sample['mean.i0.id'].append(meanI0_level)

			jumpF0_level = convert_value_to_level(sample['jump.f0'][index], semitone_bins)
			sample['jump.f0.id'].append(jumpF0_level)

			jumpI0_level = convert_value_to_level(sample['jump.i0'][index], semitone_bins)
			sample['jump.i0.id'].append(jumpI0_level)

			rangeF0_level = convert_value_to_level(sample['range.f0'][index], semitone_bins)
			sample['range.f0.id'].append(rangeF0_level)

			rangeI0_level = convert_value_to_level(sample['range.i0'][index], semitone_bins)
			sample['range.i0.id'].append(rangeI0_level)

			slopeF0_level = convert_value_to_level(sample['slope.f0'][index], slope_bins)
			sample['slope.f0.id'].append(slopeF0_level)


def create_pause_bins():
	bins = np.arange(0, 1, 0.05)
	bins = np.concatenate((bins, np.arange(1, 2, 0.1)))
	bins = np.concatenate((bins, np.arange(2, 5, 0.2)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def create_semitone_bins():
	bins = np.arange(-20, -10, 1)
	bins = np.concatenate((bins, np.arange(-10, -5, 0.5)))
	bins = np.concatenate((bins, np.arange(-5, 0, 0.25)))
	bins = np.concatenate((bins, np.arange(0, 5, 0.25)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def create_slope_bins():
	bins = np.arange(-100, -50, 10)
	bins = np.concatenate((bins, np.arange(-50, 50, 1)))
	bins = np.concatenate((bins, np.arange(50, 101, 10)))
	return bins

def main(options):
	if not checkArgument(options.input_dir, isDir=True):
		sys.exit("Input directory missing")
	if not checkArgument(options.output_dir, isDir=True, createDir=True):
		sys.exit("Output directory missing")
	else:
		ALL_FILE = os.path.join(options.output_dir, "all")
		TRAIN_FILE = os.path.join(options.output_dir, "train")
		TEST_FILE = os.path.join(options.output_dir, "test")
		DEV_FILE = os.path.join(options.output_dir, "dev")
		WORD_VOCAB_FILE = os.path.join(options.output_dir, "vocabulary")
		LOG_FILE = os.path.join(options.output_dir, "log.txt")
		TEST_TEXT_FILE = os.path.join(options.output_dir, "test.txt")
		TEST_TEXT_PLAIN_FILE = os.path.join(options.output_dir, "test_plain.txt")
	if checkArgument(options.set_size):
		max_set_size = options.set_size
	else:
		max_set_size = 9999999999999


	logfile = open(LOG_FILE, 'wb')
	
	talkfiles = glob.glob(options.input_dir + '/*.pickle')
	talkfiles.sort()

	#merge all data in files while segmenting to samples of size MAX_SEQUENCE_LEN. Each sample should start with a new sentence. Stop when set_size is reached 
	[all_samples, extraction_data] = sample_data_from_files(talkfiles, max_set_size)

	word_counts = get_word_counts_from_samples(all_samples)
	vocabulary = build_vocabulary(word_counts, options.min_vocab)

	write_vocabulary(vocabulary, WORD_VOCAB_FILE)
	word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
	print("Vocabulary extracted to %s. (Size: %i)"%(WORD_VOCAB_FILE, len(word_vocabulary)))
	logfile.write("Vocabulary extracted to %s. (Size: %i)\n"%(WORD_VOCAB_FILE, len(word_vocabulary)))

	add_word_ids_to_samples(all_samples, word_vocabulary)
	
	add_levels_to_samples(all_samples, logfile)
	
	[all_set, training_set, testing_set, dev_set, testing_start, testing_end] = split_data_to_sets(all_samples, options.train_split_ratio)

	#write data to pickle files. Test data also to a text file
	with open(ALL_FILE, 'wb') as f:
		cPickle.dump(all_set, f, cPickle.HIGHEST_PROTOCOL)
		print("All samples dumped to %s (Size: %i)"%(ALL_FILE, len(all_set)))
		logfile.write("All samples dumped to %s (Size: %i)\n"%(ALL_FILE, len(all_set)))
	with open(TRAIN_FILE, 'wb') as f:
		cPickle.dump(training_set, f, cPickle.HIGHEST_PROTOCOL)
		print("Training samples dumped to %s (Size: %i)"%(TRAIN_FILE, len(training_set)))
		logfile.write("Training samples dumped to %s (Size: %i)\n"%(TRAIN_FILE, len(training_set)))
	with open(DEV_FILE, 'wb') as f:
		cPickle.dump(dev_set, f, cPickle.HIGHEST_PROTOCOL)
		print("Development samples dumped to %s (Size: %i)"%(DEV_FILE, len(dev_set)))
		logfile.write("Development samples dumped to %s (Size: %i)\n"%(DEV_FILE, len(dev_set)))
	with open(TEST_FILE, 'wb') as f:
		cPickle.dump(testing_set, f, cPickle.HIGHEST_PROTOCOL)
		print("Testing samples dumped to %s (Size: %i)"%(TEST_FILE, len(testing_set)))
		logfile.write("Testing samples dumped to %s (Size: %i)\n"%(TEST_FILE, len(testing_set)))
	if options.write_test_text:
		with open(TEST_TEXT_FILE, 'wb') as f1, open(TEST_TEXT_PLAIN_FILE, 'wb') as f2:
			[talks_for_test, no_of_words] = write_test_text_data(f1, f2, extraction_data, testing_start, testing_end)
			print("Test text data dumped to %s (No of words:%i)"%(TEST_TEXT_FILE, no_of_words))
			logfile.write("Test text data dumped to %s (No of words:%i)\n"%(TEST_TEXT_FILE, no_of_words))
			print("Plain test text data dumped to %s (No of words:%i)"%(TEST_TEXT_PLAIN_FILE, no_of_words))
			logfile.write("Plain test text data dumped to %s (No of words:%i)\n"%(TEST_TEXT_PLAIN_FILE, no_of_words))
			print("Talks assigned as testing data:\n%s to %s"%(talks_for_test[0], talks_for_test[-1]))
			logfile.write("Talks assigned as testing data:\n%s to %s\n"%(talks_for_test[0], talks_for_test[-1]))

	logfile.close()


if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-i", "--input_dir", dest="input_dir", default=None, help="Input directory with a pickle file for each talk", type="string")
	parser.add_option("-o", "--output_dir", dest="output_dir", default=None, help="Output directory to put train, dev, test files")
	parser.add_option("-r", "--train_percent", dest="train_split_ratio", default=0.7, help="Ratio of samples to put for training (default=0.7)", type="float")
	parser.add_option("-s", "--set_size", dest="set_size", help="number of samples to process", type="int")
	parser.add_option("-v", "--min_vocab", dest="min_vocab", default=3, help="min number of word occurence to be added into vocabulary", type="int")
	parser.add_option("-t", "--write_test_text", dest="write_test_text", default=False, help="write text data for scoring", action="store_true")

	(options, args) = parser.parse_args()
	main(options)