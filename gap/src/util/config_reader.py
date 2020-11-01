from configparser import ConfigParser
import sys, os

class Configurable(object):
	def __init__(self, config_file, extra_args=None):
		config = ConfigParser(allow_no_value=True)
		config.read(config_file)
		if extra_args:
			extra_args = dict([ (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
		for section in config.sections():
			for k, v in config.items(section):
				if k in extra_args:
					v = type(v)(extra_args[k])
					config.set(section, k, v)
		self._config = config
		if not os.path.isdir(self.output):
			os.mkdir(self.output)
		config.write(open(self.config_file,'w'))
		print('Loaded config file sucessfully.')
		for section in config.sections():
			for k, v in config.items(section):
				print(k, v)

	@property
	def external_embedding(self):
		return self._config.get('Data','external_embedding')
	@property
	def max_len(self):
		return self._config.get('Data', 'max_len')
	@property
	def data_dir(self):
		return self._config.get('Data','data_dir')
	@property
	def conll_train(self):
		return self._config.get('Data','conll_train')
	@property
	def conll_dev(self):
		return self._config.get('Data','conll_dev')
	@property
	def conll_test(self):
		return self._config.get('Data','conll_test')
	@property
	def lprop(self):
		return self._config.getfloat('Data', 'lprop')
	@property
	def uprop(self):
		return self._config.getfloat('Data', 'uprop')

	@property
	def output(self):
		return self._config.get('Save','output')
	@property
	def params(self):
		return self._config.get('Save','params')
	@property
	def savemodel(self):
		return self._config.get('Save','savemodel')
	@property
	def config_file(self):
		return self._config.get('Save','config_file')

	@property
	def wembedding_dims(self):
		return self._config.getint('Network','wembedding_dims')
	@property
	def pembedding_dims(self):
		return self._config.getint('Network','pembedding_dims')
	@property
	def hidden_units(self):
		return self._config.getint('Network','hidden_units')
	@property
	def activation(self):
		return self._config.get('Network','activation')
	@property
	def lstm_dims(self):
		return self._config.getint('Network','lstm_dims')
	@property
	def blstmFlag(self):
		return self._config.getboolean('Network', 'blstmFlag')
	@property
	def costaugFlag(self):
		return self._config.getboolean('Network','costaugFlag')

	@property
	def learning_rate(self):
		return self._config.getfloat('Optimizer','learning_rate')
	@property
	def optim(self):
		return self._config.get('Optimizer','optim')
	@property
	def clip(self):
		return self._config.getfloat('Optimizer','clip')
	@property
	def emscale(self):
		return self._config.getfloat('Optimizer', 'emscale')

	@property
	def modelname(self):
		return self._config.get('Run', 'modelname')
	@property
	def epochs(self):
		return self._config.getint('Run','epochs')
	@property
	def train_batch_size(self):
		return self._config.getint('Run','train_batch_size')
	@property
	def test_batch_size(self):
		return self._config.getint('Run','test_batch_size')
	@property
	def gpuFlag(self):
		return self._config.getboolean('Run','gpuFlag')
	@property
	def verbose(self):
		return self._config.getboolean('Run', 'verbose')

