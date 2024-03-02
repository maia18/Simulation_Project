# Simulation Monte Carlo - Cellular Wireless Communication Systems
import numpy as np ; import matplotlib.pyplot as plt ; from math import sqrt,log10,log2

class PointAcess: # Point Acess

  channel = list(range(1,101)) # Channels

  # class constructor
  def __init__(self, coverage_area:tuple, power_=0):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    assert type(coverage_area) == tuple and ( len(coverage_area) >= 0 ) # Coverage area must be an tuple with len positive
    self.__coveragearea = (coverage_area)
    self.__power = power_

  # Get coverage area
  @property
  def coverage_area(self):

    return self.__coveragearea

  # Set coverage area
  @coverage_area.setter
  def coverage_area(self, coverage_area_:tuple):
    
    assert type(coverage_area_) == tuple and ( len(coverage_area_) >= 0 ) # New Coverage area must be an tuple with len positive
    self.__coveragearea = (coverage_area_)


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

    height, width = self.coverage_area

    if 0 <= position_[0] < height and 0 <= position_[1] < width:
        self.__position = position_
        self.__coveragearea[self.__position[0]:self.__position[0] + 10, self.__position[1]:self.__position[1] + 10] = 1
        return self.__position
    
    else:
        raise AssertionError("Position out of the coverage area...")



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
  def aps(self, aps_: list[PointAcess]):

    assert isinstance(aps_, list) # AP must be an instance of the class PointAcess for be add on the list
    self.__aps.extend(aps_)


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
    self.system = system
    self.__coords = []  # Coordinates ocupeds

    self.bt = (10**(8)) # Total available bandwidth ( 100MHz = 10^(8)Hz )
    self.ko = (10**(-20)) # Constant for the noise power ( 10^(-17)miliwatts/Hz = 10^(-20)watts/Hz )
    self.do = 1 # fixed reference distance ( 1 meter )
    self.k = (10**(-4)) # Constant for the propagation model
    self.n = 4 # Constant for the propagation model
      
  def add_aps(self, num_aps: int, coverage_area: tuple, power: int, min_distance: float):

    assert num_aps > 0
    assert min_distance > 0

    for _ in range(num_aps):
      
      while True:
        
        pos_ap = (np.random.randint(0, 1000), np.random.randint(0, 1000))
            
        if (self.__coords.__contains__(pos_ap) == False):
          min_distance_satisfied = all(sqrt((pos_ap[0] - ap.position_ap[0]) ** 2 + (pos_ap[1] - ap.position_ap[1]) ** 2) >= min_distance for ap in self.system.aps)

          if min_distance_satisfied:
            ap = PointAcess(coverage_area, power)
            ap.position_ap = pos_ap
            self.system.aps.append(ap)
            self.__coords.append(pos_ap)
            break


  def UE_position(self, ue: UserEquipments, aps: list[PointAcess]): # Position UE

    assert isinstance(ue, UserEquipments) # ue must be an instance of the UserEquipments
    assert isinstance(aps, list) # aps must be an list of the PointAcess

    max_height, max_width = 0, 0

    for ap in aps:
        
        height, width = ap.coverage_area

        if (height >= max_height) and (width >= max_width):
            max_height = height
            max_width = width
        
    while True:
      
      ap = np.random.choice(aps)
          
      pos__ue = ((np.random.uniform(ap.position_ap[0] - min(max_height, max_width), ap.position_ap[0] + min(max_height, max_width)), np.random.uniform(ap.position_ap[1] - min(max_height, max_width), ap.position_ap[1] + min(max_height, max_width))))
      
      if (self.__coords.__contains__(pos__ue) == False) and (((pos__ue[0] - ap.position_ap[0]) ** 2 + (pos__ue[1] - ap.position_ap[1]) ** 2) <= (min(max_height, max_width) ** 2)):
        
        distance_ue_ap = sqrt( ( ( ( pos__ue[0]- ap.position_ap[0] ) ** 2 ) ) + ( ( ( pos__ue[1] - ap.position_ap[1] ) ** 2 ) ) ) # Distance UE-AP

        if distance_ue_ap >= self.do:  # Distance must be bigger or equal than the fixed reference distance ( 1 meter )
          
          ue.position_ue = pos__ue
          self.__coords.append(pos__ue)
          break


  def distance_ue_ap_(self, ap: PointAcess, ue: UserEquipments): # Calcule distance UE-AP
    
    if isinstance(ap, PointAcess) and isinstance(ue, UserEquipments):

      for ap_ in system.aps:  
        
        distance_ue_ap = sqrt( ( ( ( ue.position_ue[0] - ap_.position_ap[0] ) ** 2 ) ) + ( ( ( ue.position_ue[1] - ap_.position_ap[1] ) ** 2 ) ) ) # Distance UE-AP
      
        if distance_ue_ap >= self.do:  # Distance must be bigger or equal than the fixed reference distance ( 1 meter )
        
          return distance_ue_ap
      
        else:

          raise ValueError('Distance lower than 1 meter!')
      


if __name__ == "__main__":

  powers = [] # Powers totallys
  snrs = [] # snrs totallys
  sirs = [] # sirs totallys
  sinrs = [] # sinrs totallys
  capacities = [] # capacities totallys

  system = System()
  simulate = Simulation(system)

  simulate.add_aps(10, (100, 100), 10, 310)

  for i, ap in enumerate(system.aps):
      print(f"AP {i+1} - Position: {ap.position_ap}")

  noise_power = ( ( simulate.ko ) * ( simulate.bt / len( ap.channel ) ) if len( ap.channel ) >= 0 else None ) # Noise power
  print(f'Noise Power: {noise_power}W \n')

  num_ue = 100 # Amount of UEs
  ues_ = [UserEquipments() for _ in range(num_ue)]

  for ue in ues_:
    
    system.ues = ue
    
  for i in range(num_ue):
    
    simulate.UE_position(ues_[i], system.aps)

  for i, ue in enumerate(ues_):
    
    print(f"Position UE {i+1} : {ue.position_ue}") # Position UE
    print(f"UE {i+1} Channel: {ue.get_channel()}") # Channel UE 
    print(f"Distance UE{i+1}-AP  : {simulate.distance_ue_ap_(ap, ue)}m") # Distance AP-UE
      
    power = ( ue.power * ( simulate.k / ( simulate.distance_ue_ap_(ap, ue) ** ( simulate.n ) ) ) ) # Power in Watts
    
    print(f"Power UE{i+1}: {power}W") ; powers.append(power)
        
    for j, ues in enumerate(ues_):
      
      interference_ = 0

      if ( ( ues.get_channel() == ue.get_channel() ) and ( ues != ue ) ):
        
        distance_ues_ap = sqrt( ( ( ( ues.position_ue[0] - ap.position_ap[0] ) ** (2) ) + ( ( ues.position_ue[1] - ap.position_ap[1] ) ** (2) ) ) ) # Distance Others_UEs-AP
        print(f"\nDistance between UE {j+1} and AP: {distance_ues_ap}m")
        interference_ += ( ues.power *  ( ( distance_ues_ap / ( simulate.do ) ** ( simulate.n ) ) ) ) # interference totally
        print(f"Interference between UE {j+1} and AP: {interference_}\n")
      
        if interference_ >= 0:
          
          snr = ( 10 ) * log10( ( ( power / noise_power ) ) ) # SNR
          sir = ( 10 ) * log10( ( power / interference_ ) ) # SIR
          sinr = ( 10 ) * log10( ( power / ( interference_ + noise_power ) ) ) # SINR
          capacity = ( simulate.bt / len(ap.channel) ) * ( log2(1 + ( 10**(sinr/10) ) ) ) # Capacity
            
          print(f"Signal-to-noise ratio(SNR): {snr}db") ; snrs.append(snr)
          print(f"Signal-to-interference ratio(SIR): {sir}db") ; sirs.append(sir)
          print(f"Signal-to-interference-Noise ratio(SINR): {sinr}db") ; sinrs.append(sinr)
          print(f"Capacity: {capacity}") ; capacities.append(capacity) ; print("- "*80)

          all_ = []
          all_.append([powers, snrs, sirs, sinrs, capacities]) # Collect all results

  fig, axs = plt.subplots(2, 3, figsize=(18, 10)) # Graphic

  for ap in system.aps:

    height, width = ap.coverage_area
    axs[0, 0].scatter(ap.position_ap[0], ap.position_ap[1], color='red', marker=',')
    cove_area = plt.Circle(ap.position_ap, radius = min(height, width), alpha=0.2)
    axs[0, 0].add_patch(cove_area)
    axs[0, 0].set_xlim(0, 1000)
    axs[0, 0].set_ylim(0, 1000)
    axs[0, 0].set_title("Simulate")
    axs[0, 0].grid(True)

    for ue in system.ues:

        axs[0, 0].scatter(ue.position_ue[0], ue.position_ue[1], color='black', marker='.')

    for result, label, row, col in zip([powers, snrs, sirs, sinrs, capacities], ['Power', 'SNR', 'SIR', 'SINR', 'Capacity'], [0, 0, 1, 1, 1], [1, 2, 0, 1, 2]):
      
      filtered_result = [value for value in result if value is not None]
      filtered_result.sort()
      cumulative_prob = np.linspace(0, 1, len(filtered_result))
      axs[row, col].plot(filtered_result, cumulative_prob, label=f"CDF - {label}")
      axs[row, col].set_title(f"CDF - {label}")
      axs[row, col].grid(True)
  
  plt.show()