# Simulation Monte Carlo - Cellular Wireless Communication Systems
import numpy as np; import matplotlib.pyplot as plt; from math import sqrt,log10,log2,ceil; import random

class PointAcess: # Point Acess

  def __init__(self, coverage_area:tuple, power_):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    assert (type(coverage_area) == tuple) and ( len(coverage_area) >= 0 ) # Coverage area must be an tuple with len positive
    self.__coveragearea = coverage_area # Coverage Area
    self.__power = power_ # Power

  @property
  def coverage_area(self): # Get coverage area
    return self.__coveragearea

  @coverage_area.setter
  def coverage_area(self, coverage_area_:tuple): # Set coverage area
    assert type(coverage_area_) == tuple and ( len(coverage_area_) >= 0 ) # New Coverage area must be an tuple with len positive
    self.__coveragearea = (coverage_area_) # New Coverage Area

  @property
  def power(self): # Get power
    return self.__power

  @power.setter
  def power(self, power__): # Set power
    assert not isinstance(power__, (str, bool, list, tuple)) and ( power__ >= 0 ) # Power must be an number positive
    self.__power = power__ # New Power

  def position_ap(self, position_:tuple): # Position - AP
    assert isinstance(position_, tuple) and ( len(position_) >= 0 ) # Position must be an tuple with len postive
    assert (0 <= position_[0] <= self.coverage_area[0]) and (0 <= position_[1] <= self.coverage_area[1])
    self.__position = position_
    self.__coveragearea[self.__position[0]:self.__position[0] + 10, self.__position[1]:self.__position[1] + 10] = 1
    return self.__position

class UserEquipments: # User Equipments

  def __init__(self, channel=None, power_=1):

    assert (not isinstance(power_, (str, bool, list, tuple))) and (( power_ >= 0 )) # Power must be an number positive
    self.__channel = channel # Channel
    self.__power = power_ # Power

  @property
  def power(self): # Get power
    return self.__power

  @power.setter
  def power(self, power__): # Set power
    assert not isinstance(power__, (str, bool, list, tuple)) and ( power__ >= 0 ) # Power must be an number positive
    self.__power = power__

  def position_ue(self, position_:tuple): # Position - UE
    assert isinstance(position_, tuple) and ( len(position_) >= 0 ) # Position must be an tuple with len postive
    self.__position = position_
    return self.__position

  def channel(self): # Channel
    return self.__channel

class System: # System

  def __init__(self):

    self.__aps = list() # Point Acess
    self.__ues = list() # User Equipments

  @property
  def aps(self): # Get AP's
    return self.__aps
  
  @aps.setter
  def aps(self, aps_: list[PointAcess]): # Set AP's
    assert isinstance(aps_, list) # AP must be an instance of the class PointAcess for be add on the list
    self.__aps.extend(aps_)

  @property
  def ues(self): # Get UE's
    return self.__ues

  @ues.setter
  def ues(self, ues_: list[UserEquipments]): # Set UE's
    assert isinstance(ues_, list) # UE must be an instance of the class UserEquipments for be add on the list
    self.__ues.extend(ues_)

class Simulation: # Simulation

  def __init__(self, system:System, num_ues, num_aps, num_sms:int, channels=list(range(1,4))):

    assert isinstance(system, System) # system must be an instance of the class System
    self.system = system
    self.coords = [] # Coordinates ocupeds

    self.num_sms = num_sms # Amount of simulations
    self.num_aps = num_aps # Amount of APs
    self.num_ues = num_ues # Amount of UEs
    self.channels = channels # Amount of Channels

    self.noise_power = ((((10**(-20))) * ((10**(8)) / len(self.channels)))) # Noise power
    self.bt = (10**(8)) # Total available bandwidth ( 100MHz = 10^(8)Hz )
    self.ko = (10**(-20)) # Constant for the noise power ( 10^(-17)miliwatts/Hz = 10^(-20)watts/Hz )
    self.k = (10**(-4)) # Constant for the propagation model
    self.n = 4 # Constant for the propagation model

  def distance(self, ues:list[UserEquipments], aps:list[PointAcess]): # Distance UE-AP
    distances_ = {} # Dict of distances
    distances_min = {} # Dict of distances min

    for _, ue in enumerate(ues): # Ues
      for __, ap in enumerate(aps): # Aps
        distance_ue_ap = sqrt((ue.position_ue[0] - ap.position_ap[0]) ** 2 + (ue.position_ue[1] - ap.position_ap[1]) ** 2) # Distance ue-ap
        if (distance_ue_ap) >= 1 and distance_ue_ap not in distances_: # Avoid sames distances
            distances_[f'UE {_+1} - AP {__+1}'] = distance_ue_ap # Add the distance on the list of distances
    for key, value in distances_.items(): # Acess the Dict of distances min
      ue = key.split(' - ')[0] # Number of UE
      if ue not in distances_min: # Avoid sames ues
          distances_min[ue] = value # Distance min
      else:
          if value < distances_min[ue]: # Avoid sames Distance min
              distances_min[ue] = value
    return list(distances_min.values()) # Return a list of distances min

  def AP_position(self, aps: list[PointAcess]): # Position AP
    assert (len(aps) > 0) and isinstance(aps, list) # Amount of APs must be bigger than zero
    for i in range(len(aps)):
        while True:
          pos_ap = (((i%int(sqrt(len(aps)))) + 0.5) * 1000 / (int(sqrt(len(aps)))), ((i//int(sqrt(len(aps)))) + 0.5) * 1000 / ceil(len(aps) / int(sqrt(len(aps)))))
          if self.coords.__contains__(pos_ap) == False:
            aps[i].position_ap = pos_ap
            self.coords.append(aps[i].position_ap)
            break

  def UE_position(self, ues: list[UserEquipments]): # Position UE
    assert isinstance(ues, list) # ues must be an list of the PointAcess
    for i in range(len(ues)):
      while True:
        pos_ue = (np.random.randint(0, 1000), np.random.randint(0, 1000))
        if (self.coords.__contains__(pos_ue) == False):
          ues[i].position_ue = pos_ue
          self.coords.append(ues[i].position_ue)
          break

  def load_results(self, file_path): # Load results of the simulation previou
    data = np.load(file_path)
    return data['sinrs'], data['cdf_sinrs'], data['capacities'], data['cdf_capacities']

  def run_simulation(self, save_file=None, load_file=None): # Run Simulation
    
    if load_file: # Load results
      sinrs_sorted, cdf_sinrs, capacities_sorted, cdf_capacities = self.load_results(load_file)
    else: # Simulation unprecedented
      system.aps = [PointAcess((1000, 1000), 10) for i in range(self.num_aps)] # APs
      system.ues = [UserEquipments(np.random.choice(self.channels)) for j in range(self.num_ues)] # UEs'
      self.AP_position(system.aps) # Position APs

      # List of the results totallys
      sinrs_totallys = []
      capacities_totallys = []

      # Simulation  
      for _ in range(self.num_sms):
        sinrs = [] # List of SINRS results
        capacities = [] # List of CAPACITY results
        self.UE_position(system.ues) # Position UEs
        interference_ = 0
        for distance__ in self.distance(system.ues, system.aps):
          power = (UserEquipments().power * (self.k / (distance__ ** (4)))) # Power in Watts
        for ue in system.ues:
          other_ues_ = [other_ue for other_ue in system.ues if other_ue != ue]
        for _, other_ue in enumerate(other_ues_):
          if other_ue.channel() == ue.channel():
            interference_ += (((other_ue.power * (self.k / ((self.distance(other_ues_, system.aps)[_]) ** (4))))))
            if interference_ > 0:
              sinr = ((power / (interference_ + self.noise_power))) ; sinr_db = 10 * log10(sinr) ; sinrs.append(sinr_db) # SINR
              capacity = ((self.bt / len(self.channels)) * (log2(1 + sinr))) ; capacities.append(capacity) # Capacity

        sinrs_totallys.extend(sinrs)
        capacities_totallys.extend(capacities)
        sinrs_sorted = sorted(sinrs_totallys)
        capacities_sorted = sorted(capacities_totallys)
        cdf_sinrs = np.linspace(0, 1, len(sinrs_sorted))
        cdf_capacities = np.linspace(0, 1, len(capacities_sorted))

      if save_file: # Save results
        np.savez(save_file, sinrs=sinrs_sorted, cdf_sinrs=cdf_sinrs, capacities=capacities_sorted, cdf_capacities=cdf_capacities)     
    return sinrs_sorted, cdf_sinrs, capacities_sorted, cdf_capacities # Return results

if __name__ == "__main__":
  
  system = System() # System with ues, aps.
  
  all_sinrs = [] # All SINRs
  all_cdf_sinrs = [] # All CDF of SINRS
  all_capacities = [] # All CAPACITIES
  all_cdf_capacities = [] # All CDF of CAPACITIES

  max_test = 5 # Quantify max of tests

  aps = [1, 4, 9, 16, 25, 36, 49, 64] # Amount of APs
  ues = [4, 8, 10, 12, 15, 20, 25, 50] # Amount of UEs
  channels = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] # Amount of channels

  fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # Graphic

  comb = [] # List of combinations ue-ap-channel
  for i in range(1, max_test):
    ue, ap, ch = random.choice(ues),random.choice(aps),random.choice(channels)
    combs = [ue, ap, ch]
    if combs not in comb: # Avoid sames combinations
      comb.append(combs)

      simulate = Simulation(system, ue, ap, 100, ch)
      simulate.run_simulation(save_file='results.npz')
      sinrs, cdf_sinrs, capacities, cdf_capacities = simulate.run_simulation(load_file="results.npz")
        
      all_sinrs.append(sinrs)
      all_cdf_sinrs.append(cdf_sinrs)
      all_capacities.append(capacities)
      all_cdf_capacities.append(cdf_capacities)

      axs[0].plot(all_sinrs[-1], all_cdf_sinrs[-1], label=f'{ue} UEs, {ap} APs, {len(ch)} Channels') 
      axs[1].plot(all_capacities[-1], all_cdf_capacities[-1], label=f'{ue} UEs, {ap} APs,{len(ch)} Channels')

axs[0].set_title('CDF - SINR')
axs[0].grid(True)
axs[0].legend(fontsize=6.5)
axs[1].set_title('CDF - Capacity')
axs[1].grid(True)
axs[1].legend(fontsize=6.5)
plt.show()