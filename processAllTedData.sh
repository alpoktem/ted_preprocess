database_dir=$1
output_dir=$2

f0_dir=$database_dir/derived/segs/f0
i0_dir=$database_dir/derived/segs/i0
txt_sent_dir=$database_dir/txt-sent

mkdir -p $output_dir

fileNo=0
total_no_of_samples=0
for file_wordalign in `ls $txt_sent_dir | grep "word.txt.norm.align"`; do
	talk_id=`echo $file_wordalign | cut -d. -f1`
	echo $talk_id
	file_word="$txt_sent_dir/$talk_id.word.txt"
	file_wordalign="$txt_sent_dir/$talk_id.word.txt.norm.align"
	file_wordaggs_f0="$f0_dir/$talk_id.aggs.alignword.txt"
	file_wordaggs_i0="$i0_dir/$talk_id.aggs.alignword.txt"

	python tedDataToPickle.py -w $file_word -l $file_wordalign  -f $file_wordaggs_f0 -i $file_wordaggs_i0 -o $output_dir/$talk_id.pickle -c $output_dir/$talk_id.csv
	
done