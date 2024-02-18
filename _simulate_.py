# Simulation Monte Carlo - Cellular Wireless Communication Systems
import numpy as np ; import matplotlib.pyplot as plt ; from math import sqrt,log10,log2

class PointAcess: # Point Acess

  channel = list(range(1,101)) # Channels

  # class constructor
  def __init__(self, coverage_area=(0,0), power_=0):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    assert type(coverage_area) == tuple and ( len(coverage_area) >= 0 ) # Coverage area must be an tuple with len positive
    self.__coveragearea = np.zeros(coverage_area)
    self.__power = power_

  # Get coverage area
  @property
  def coverage_area(self):

    return self.__coveragearea

  # Set coverage area
  @coverage_area.setter
  def coverage_area(self, new_coverage_area_:tuple):
    
    assert type(new_coverage_area_) == tuple and ( len(new_coverage_area_) >= 0 ) # New Coverage area must be an tuple with len positive
    self.__coveragearea = np.zeros(new_coverage_area_)


  # Get power
  @property
  def power(self):

    return self.__power

  # Set power
  @power.setter
  def power(self, power__):
    
    assert not isinstance(power__, (str, bool, list, tuple)) and ( power__ >= 0 ) # Power must be an number positive
    self.__power = power__


  def position_ap(self, position_:tuple): # Position - AP
    
    assert isinstance(position_, tuple) and ( len(position_) >= 0 ) # Position must be an tuple with len postive
    self.__coveragearea[position_[0]:position_[0] + 10, position_[1]:position_[1]+10] = 1



class UserEquipments: # User Equipments

  # class constructor
  def __init__(self, power_=1):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    self.__channel = np.random.choice(PointAcess.channel) # Choose an channel
    self.__power = power_

  # Get power
  @property
  def power(self):

    return self.__power

  # Set power
  @power.setter
  def power(self, power__):
    
    assert not isinstance(power__, (str, bool, list, tuple)) and ( power__ >= 0 ) # Power must be an number positive
    self.__power = power__


  def position_ue(self, position_:tuple): # Position - UE
    
    assert isinstance(position_, tuple) and ( len(position_) >= 0 ) # Position must be an tuple with len postive
    self.__position = position_
    return self.__position


  # Get channel
  def get_channel(self):

    return self.__channel
  

  
class System: # System

  # class constructor
  def __init__(self):

    self.__aps = list()  # Point Acess
    self.__ues = list()  # User Equipments

  # Get AP's
  @property
  def aps(self):

    return self.__aps

  # Set AP's
  @aps.setter
  def aps(self, aps_: PointAcess):

    assert isinstance(aps_, PointAcess) # AP must be an instance of the class PointAcess for be add on the list
    self.__aps.append(aps_)


  # Get UE's
  @property
  def ues(self):

    return self.__ues

  # Set UE's
  @ues.setter
  def ues(self, ues: UserEquipments):

    assert isinstance(ues, UserEquipments) # UE must be an instance of the class UserEquipments for be add on the list
    self.__ues.append(ues)



class Simulation: # Simulation

  # class constructor
  def __init__(self, system:System):

    assert isinstance(system, System)  # system must be an instance of the class System
    self.__coords = []  # Coordinates ocupeds

    self.bt = (10**(8)) # Total available bandwidth ( 100MHz = 10^(8)Hz )
    self.ko = (10**(-20)) # Constant for the noise power ( 10^(-17)miliwatts/Hz = 10^(-20)watts/Hz )
    self.do = 1 # fixed reference distance ( 1 meter )
    self.k = (10**(-4)) # Constant for the propagation model
    self.n = 4 # Constant for the propagation model
      
  def position_AP(self, ap: PointAcess, pos__ap: tuple): # Position AP

    assert isinstance(ap, PointAcess) # AP must be an instance of the class PointAcess
    assert isinstance(pos__ap, tuple) and ( len(pos__ap) >= 0 ) # Position must be an tuple with len positive
      
    if (self.__coords.__contains__(pos__ap) == False):
        
      ap.position_ap = pos__ap
      self.__coords.append(pos__ap)

    else:
          
      while True:
        
        pos__ap_ = ((np.random.randint(0, ...), np.random.randint(0, ...)))

        if self.__coords.__contains__(pos__ap_) == False:
          
          ap.position_ap = pos__ap_
          self.__coords.append(pos__ap_)
          break


  def position_UE(self, ue: UserEquipments, ap:PointAcess, pos__ue=(0,0)): # Position UE
    
    radius = len(ap.coverage_area)

    if isinstance(ue, UserEquipments) and isinstance(ap, PointAcess) and isinstance(pos__ue, tuple) and len(pos__ue) >= 0:
      
      if self.__coords.__contains__(pos__ue) == False and (((pos__ue[0] - ap.position_ap[0]) ** 2 + (pos__ue[1] - ap.position_ap[1]) ** 2) <= (radius ** 2)):
        
        ue.position_ue = pos__ue
        self.__coords.append(pos__ue)

      else:
        
        while True:
          
          pos__ue = ((np.random.uniform(ap.position_ap[0] - radius, ap.position_ap[0] + radius), np.random.uniform(ap.position_ap[1] - radius, ap.position_ap[1] + radius)))
          
          if self.__coords.__contains__(pos__ue) == False and (((pos__ue[0] - ap.position_ap[0]) ** 2 + (pos__ue[1] - ap.position_ap[1]) ** 2) <= (radius ** 2)):
            
            ue.position_ue = pos__ue
            self.__coords.append(pos__ue)
            break


  def distance_ue_ap_(self, ap: PointAcess, ue: UserEquipments): # Calcule distance UE-AP
    
    if isinstance(ap, PointAcess) and isinstance(ue, UserEquipments):
      
      distance_ue_ap = sqrt( ( ( ( ue.position_ue[0] - ap.position_ap[0] ) ** 2 ) ) + ( ( ( ue.position_ue[1] - ap.position_ap[1] ) ** 2 ) ) ) # Distance UE-AP
      
      if distance_ue_ap >= self.do:  # Distance must be bigger or equal than the fixed reference distance ( 1 meter )
        
        return distance_ue_ap
      


if __name__ == "__main__":

  powers = [] # Powers totallys
  snrs = [] # snrs totallys
  sirs = [] # sirs totallys
  sinrs = [] # sinrs totallys
  capacities = [] # capacities totallys

  AP = PointAcess((1000,1000), 10)

  system = System()
  system.aps = AP

  simulate = Simulation(system)
  simulate.position_AP(AP, (0,0))

  num_ue = 100 # Amount of UEs
  ues_ = [UserEquipments() for _ in range(num_ue)]

  for ue in ues_:
    
    system.ues = ue
    
  for i in range(num_ue):
    
    simulate.position_UE(ues_[i], AP)

  for i, ue in enumerate(ues_):
    
    print(f"Position UE {i+1} : {ue.position_ue}") # Position UE
    print(f"UE {i+1} Channel: {ue.get_channel()}") # Channel UE 
    print(f"Distance AP-UE {i+1} : {simulate.distance_ue_ap_(AP, ue)}m") # Distance AP-UE
      
    for j, ues in enumerate(ues_): 
      
      if ( ( ues.get_channel() == ue.get_channel() ) and ( ues != ue ) ):
        
        distance_ues = sqrt( ( ( ( ues.position_ue[0] - ue.position_ue[0] ) ** (2) ) + ( ( ues.position_ue[1] - ue.position_ue[1] ) ** (2) ) ) ) # Distance UE-Others_UEs
        print(f"\nDistance between UE {i+1} and UE {j+1} : {distance_ues}m")
        interference_ = 0
        interference_ += ( AP.power *  ( ( distance_ues / ( simulate.do ) ** ( simulate.n ) ) ) ) # interference totally
        print(f"Interference between UE {i+1} and UE {j+1} : {interference_}\n")
      
        if ( interference_ >= 0 ):
          
          power = ( ue.power * ( simulate.k / ( simulate.distance_ue_ap_(AP, ue) ** ( simulate.n ) ) ) ) # Power in Watts
          noise_power = ( ( simulate.ko ) * ( simulate.bt / len( AP.channel ) ) if len( AP.channel ) >= 0 else None ) # Noise power

          if ( power > 0 ) and ( noise_power > 0 ):
            
            snr = ( 10 ) * log10( ( ( power / noise_power ) ) ) # SNR
            sir = ( 10 ) * log10( ( power / interference_ ) ) # SIR
            sinr = ( 10 ) * log10( ( power / ( interference_ + noise_power ) ) ) # SINR
            capacity = ( simulate.bt / len(AP.channel) ) * ( log2(1 + ( 10**(sinr/10) ) ) ) # Capacity
              
            print(f"Power: {power}W") ; powers.append(power)
            print(f"Signal-to-noise ratio(SNR): {snr}db") ; snrs.append(snr)
            print(f"Signal-to-interference ratio(SIR): {sir}db") ; sirs.append(sir)
            print(f"Signal-to-interference-Noise ratio(SINR): {sinr}db") ; sinrs.append(sinr)
            print(f"Capacity: {capacity}") ; capacities.append(capacity) ; print("- "*80)

            all_ = []
            all_.append([powers, snrs, sirs, sinrs, capacities]) # Collect all results

  fig, axs = plt.subplots(2, 3, figsize=(18, 10)) # Graphic

  for ap in system.aps:

    axs[0, 0].scatter(ap.position_ap[0], ap.position_ap[1], color='red', marker=',')
    cove_area = plt.Circle(ap.position_ap, radius=len(ap.coverage_area), alpha=0.2)
    axs[0, 0].add_patch(cove_area) ; axs[0, 0].set_xlim(-1000, 1000) ; axs[0, 0].set_ylim(-1000, 1000) ; axs[0,0].set_title("Simulate")

    for ue in system.ues:

        axs[0, 0].scatter(ue.position_ue[0], ue.position_ue[1], color='black', marker='.')

    for result, label, row, col in zip([powers, snrs, sirs, sinrs, capacities], ['Power', 'SNR', 'SIR', 'SINR', 'Capacity'], [0, 0, 1, 1, 1], [1, 2, 0, 1, 2]):
      
      filtered_result = [value for value in result if value is not None] ; filtered_result.sort()

      cumulative_prob = np.linspace(0, 1, len(filtered_result))
      axs[row, col].plot(filtered_result, cumulative_prob, label=f"CDF - {label}") ; axs[row, col].set_title(f"CDF - {label}")

  plt.show()