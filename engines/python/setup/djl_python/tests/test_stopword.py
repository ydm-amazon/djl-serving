import unittest
from djl_python.huggingface import HuggingFaceService, StopWord
from djl_python.seq_scheduler.lm_block import HuggingfaceBlock, BloomBlock, FalconBlock

class TestStopword(unittest.TestCase):
	def setUp(self):
		self._service = HuggingFaceService()
		# Set self._service.tokenizer to a particular tokenizer
		# Perhaps I need to run _service.initialize too, but that
		# would require a serving.properties file
		# That would be closer to an integration test
		# Maybe I should just load in a tokenizer from HuggingFace
		# so that the tests are more certain
		# Actually that's probably the best strategy
		# so just set self._service.tokenizer to some AutoTokenizer
		# or even specify a tokenizer

	def test_parse_ss_input(self):  
		self.parse_ss_input_helper('["apple", "banana"]', ["apple", "banana"])
		self.parse_ss_input_helper('["<User>", "See you later"]', ["<User>", "See you later"])

	def parse_ss_input_helper(self, stop_sequence, ref):    
		"""
		Input: stop_sequence: str
		Input: ref: list of strings
		"""
		fn_output = self._service.parse_stop_sequence_input(stop_sequence)
		assert len(fn_output) == len(ref)
		for i in range(len(fn_output)):
			assert fn_output[i] == ref[i]

	def test_load_sc_list(self):
		"""
		Just tests the length of the list for now
		Maybe I can verify for each element the StopWord.stop_seq too
		"""
		stop_sequence = '["<User>", "See you later"]'
		parsed = self._service.parse_stop_sequence_input(stop_sequence)
		# error is because self.tokenizer is none
		# define a tokenizer first and this will help with other tests too
		# to do that, need to mock up inputs.get_properties() (line 640 of huggingface.py)
		# that needs to be inspected when running huggingface.py on ec2 gpu
		# Remember how to run with ec2 gpu
		self._service.load_stopping_criteria_list(stop_sequence)
		print("parsed: ", parsed)
		print("self._service.stopping_criteria_list: ", self._service.stopping_criteria_list)
		assert len(parsed) == len(self._service.stopping_criteria_list)

	# explore all test cases (both paths of if)
	# security ?? i
	def test_stopword_call(self):
		# Create a scenario with a particular tokenizer
		# and determine the exact outputs of that tokenizer on some stopword sequence
		# create a torch LongTensor
		# 
		# 
		# ...
		assert True
