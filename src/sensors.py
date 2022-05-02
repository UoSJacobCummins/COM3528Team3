#from abc import ABC

class Sensor(object):
	def __init__(self, ego_angle, environment):
		# Initializes the sensor.
		# ego_angle: Angular position of the sensor in the body
		# environment: An instance of the enviroment to get the redings from
		self.pos = [0,0]
		self.ego_angle = ego_angle
		self.environment = environment

	def setPosition( self, x, y ):
		self.pos = [x,y]

	def getReading( self ):
		pass

	def clone( self ):
		s = Sensor( self.ego_angle, self.environment )
		s.pos = [self.pos[0], self.pos[1]]
		return s

class TemperatureSensor( Sensor ):
	def __init__( self, ego_angle, environment ):
		super(TemperatureSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		return self.environment.getTemperature( self.pos[0], self.pos[1] )

class ChemicalSensor( Sensor ):
	def __init__( self, ego_angle, environment ):
		super(ChemicalSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		return self.environment.getFoodSignal( self.pos[0], self.pos[1] )

# -----------------------------------------------
# MIRO sensors
# -----------------------------------------------

class MiroTemperatureSensor( Sensor ):
	def __init__( self, ego_angle, environment):
		super(MiroTemperatureSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		hemi = 'left' if self.ego_angle < 0 else 'right'
		return self.environment.getTemperature( hemi )

class MiroChemicalSensor( Sensor ):
	def __init__( self, ego_angle, environment ):
		super(MiroChemicalSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		hemi = 'left' if self.ego_angle < 0 else 'right'
		return self.environment.getFoodSignal( hemi )
