class Struct(dict):
	"""
	Struct inherits from dict and adds this functionality:
	    Instead of accessing the keys of struct by typing
		struct['key'], one may instead type struct.key.
	These two options will do exactly the same thing. A new
	Struct object can also be created with a dict as an input
	parameter, and the resulting Struct object will have the
	same data members as the dict passed to it.
	"""
	def __init__(self,inpt={}):
		super(Struct,self).__init__(inpt)

	def __getattr__(self, name):
		return self.__getitem__(name)

	def __setattr__(self,name,value):
		self.__setitem__(name,value)
