# Simulation Monte Carlo - Cellular Wireless Communication Systems
import numpy as np;import matplotlib.pyplot as plt;from math import sqrt,log10,log2,ceil;

class PointAcess: # Point Acess

  channel = list(range(1,4)) # Channels
  noise_power = ((((10**(-20))) * ((10**(8)) / len(channel)))) # Noise power

  # class constructor
  def __init__(self, coverage_area:tuple, power_):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    assert (type(coverage_area) == tuple) and ( len(coverage_area) >= 0 ) # Coverage area must be an tuple with len positive
    self.__coveragearea = (coverage_area) # Coverage Area of AP
    self.__power = power_ # Power of AP

  # Get coverage area
  @property
  def coverage_area(self):

    return self.__coveragearea

  # Set coverage area
  @coverage_area.setter
  def coverage_area(self, coverage_area_:tuple):

    assert type(coverage_area_) == tuple and ( len(coverage_area_) >= 0 ) # New Coverage area must be an tuple with len positive
    self.__coveragearea = (coverage_area_) # New Coverage Area


  # Get power
  @property
  def power(self):

    return self.__power

  # Set power
  @power.setter
  def power(self, power__):

    assert not isinstance(power__, (str, bool, list, tuple)) and ( power__ >= 0 ) # Power must be an number positive
    self.__power = power__ # New Power


  def position_ap(self, position_:tuple): # Position - AP

    assert isinstance(position_, tuple) and ( len(position_) >= 0 ) # Position must be an tuple with len postive
    assert (0 <= position_[0] <= self.coverage_area[0]) and (0 <= position_[1] <= self.coverage_area[1])
    self.__position = position_
    self.__coveragearea[self.__position[0]:self.__position[0] + 10, self.__position[1]:self.__position[1] + 10] = 1
    return self.__position

class UserEquipments: # User Equipments

  # class constructor
  def __init__(self, power_=1):

    assert not isinstance(power_, (str, bool, list, tuple)) and ( power_ >= 0 ) # Power must be an number positive
    self.__channel = np.random.choice(PointAcess.channel) # Choose an channel
    self.__power = power_ # Power of UE

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
  def ues(self, ues_: list[UserEquipments]):

    assert isinstance(ues_, list) # UE must be an instance of the class UserEquipments for be add on the list
    self.__ues.extend(ues_)


  def distance_min(self, ue, ap):

    distances_ue_ap_min = [sqrt((ue.position_ue[0] - ap.position_ap[0]) ** 2 + (ue.position_ue[1] - ap.position_ap[1]) ** 2) for ap in self.__aps]
    return min(distances_ue_ap_min)

class Simulation: # Simulation

  # class constructor
  def __init__(self, system:System, num_aps:int, num_sms:int):

    assert isinstance(system, System) # system must be an instance of the class System
    assert (num_sms > 0) and (num_aps > 0) # Amount of simulations, aps and ues must be bigger than 0
    self.system = system
    self.coords = [] # Coordinates ocupeds

    self.num_sms = num_sms # Amount of simulations
    self.num_aps = num_aps # Amount of APs
    self.num_ues = list(range(1,6)) # Amount of UEs

    self.bt = (10**(8)) # Total available bandwidth ( 100MHz = 10^(8)Hz )
    self.ko = (10**(-20)) # Constant for the noise power ( 10^(-17)miliwatts/Hz = 10^(-20)watts/Hz )
    self.do = 1 # fixed reference distance ( 1 meter )
    self.k = (10**(-4)) # Constant for the propagation model
    self.n = 4 # Constant for the propagation model

  def distance(self, ue, ap):

    if isinstance(ue, UserEquipments):

      distance_ue_ap = sqrt((ue.position_ue[0] - ap.position_ap[0]) ** 2 + (ue.position_ue[1] - ap.position_ap[1]) ** 2)
      if (distance_ue_ap >= self.do):
        return distance_ue_ap

    if isinstance(ue, tuple):

      distance_ue_ap = sqrt((ue[0] - ap.position_ap[0]) ** 2 + (ue[1] - ap.position_ap[1]) ** 2)
      if (distance_ue_ap >= self.do):
        return distance_ue_ap



  def AP_position(self, aps: list[PointAcess]):

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


def run_simulation(simulate: Simulation):

  system.aps = [PointAcess((1000, 1000), 10) for _ in range(simulate.num_aps)]
  system.ues = [UserEquipments() for _, _ in enumerate(simulate.num_ues)]
  simulate.AP_position(system.aps)

  for _ in range(simulate.num_sms):
    sinrs_totallys = []
    capacities_totallys = []
    results_ues = []

    simulate.UE_position(system.ues)
    for j, ap in enumerate(system.aps):
      for i, ue in enumerate(system.ues):
        if simulate.distance(ue, ap) == system.distance_min(ue, ap):
          power = (ue.power * (simulate.k / (simulate.distance(ue, ap) ** (simulate.n)))) # Power in Watts
          interference_ = 0
          for k_, others_ues in enumerate(system.ues):
            if ((others_ues.get_channel() == ue.get_channel()) and (others_ues != ue)):
              interference_ += (((others_ues.power * (simulate.k / (simulate.distance(others_ues, ap) ** (simulate.n))))))  # interference totally
              if interference_ > 0:

                sinr = ((power / (interference_ + PointAcess.noise_power)))
                sinr_db = 10 * log10(sinr)
                capacity = ((simulate.bt / len(PointAcess.channel)) * (log2(1 + ((power / (interference_ + PointAcess.noise_power))))))  # Capacity in bps
                capacities_totallys.append(capacity)
                sinrs_totallys.append(sinr_db)
                results_ues.append([sinrs_totallys,capacities_totallys])

  fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                     
  sinrs_sorted = sorted(sinrs_totallys)
  capacities_sorted = sorted(capacities_totallys)
  cdf_sinrs = np.linspace(0, 1, len(sinrs_sorted))
  cdf_capacities = np.linspace(0, 1, len(capacities_sorted))

  axs[0].plot(sinrs_sorted, cdf_sinrs, label=f'{simulate.num_ues} UEs')
  axs[1].plot(capacities_sorted, cdf_capacities, label=f'{simulate.num_ues} UEs')

  axs[0].set_title('CDF of SINR')
  axs[1].set_title('CDF of Capacity')     
  plt.show()

if __name__ == "__main__":

  system = System()
  simulate = Simulation(system, 1, 1)
  
  run_simulation(simulate)