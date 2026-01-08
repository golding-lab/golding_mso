# GoldingMSO 

![ITD Visualization](/readme_files/160106_04_itd_resize.gif) 

Tools and files from the <a href="https://goldinglab.org">Golding Lab</a> at UT Austin for compartmental modeling MSO neurons, using the simulation software, <a href="https://www.neuron.yale.edu/neuron/">NEURON</a>

<br>

<a href="https://golding-mso.readthedocs.io/en/latest/"> <img alt="Static Badge" src="https://img.shields.io/badge/Read%20the%20Docs-blue?style=for-the-badge&logo=readthedocs&logoColor=white&labelColor=blue"> </a>



## Features

- 40 reconstructed MSO neuron morphologies
  - [archived](https://github.com/golding-lab/golding_mso/blob/main/cells.zip)
  - [in repo](https://github.com/golding-lab/golding_mso/blob/main/golding_mso/cells)
- Pre-defined channel definitions (KLT, KHT, HCN, Na)
  - [archived](https://github.com/golding-lab/golding_mso/blob/main/mechanisms.zip)
  - [in repo](https://github.com/golding-lab/golding_mso/blob/main/golding_mso/mechanisms)
- Pre-configured tools
  - Synapse placement
  - EPSP propagation
  - ITD procedure
  - Morphological/physiological measurements

## Getting Started

### Prerequisites

*Only tested on macOS and Windows Subsystem for Linux*

- Ensure Python 3.8+ is installed.

### Installation

Clone the repository:
```bash
git clone https://github.com/golding-lab/golding_mso.git
cd golding_mso
pip install .
```

## Resources
- [Documentation](https://golding_mso.readthedocs.io/en/latest/)
- [Golding Lab Website](https://goldinglab.org)
- [NEURON Documentation](https://www.neuron.yale.edu/neuron/)
- [GitHub Repository](https://github.com/golding-lab/golding_mso)
- [Morphology files](https://github.com/golding-lab/golding_mso/blob/main/golding_mso/cells)

<br>


## Demo
<details> 
<summary> </summary>
        

### Loading cells

```python
import golding_mso as gmso

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
```

```
516 lines read
```


### Reconstructed morphologies



```python
import golding_mso as gmso

print("Morphology cell keys:")
print("-----------------------")
for i in range(0, len(gmso.morphologies), 4):
    print(list(gmso.morphologies.keys())[i:i+4])
```

```
Morphology cell keys:
-----------------------
['151124_03', '151124_10P', '151124_11P', '151201_02_LOOK']
['151201_05_LOOK', '151201_06P', '151209_03P', '151209_06_LOOK']
['151209_09O', '151210_02P', '151210_03P_LOOK', '151210_04P_LOOK']
['151214_03', '151210_07O_LOOK', '151214_09p', '151214_10P']
['151217_03P_LOOK', '151217_04P', '151217_12p', '160105_10']
['160105_12P', '160105_14P', '160105_15P', '160106_03P_LOOK']
['160106_04_LOOK', '160111_02P', '160112_16P', '160112_19P']
['160112_20P_LOOK', '160112_26P_LOOK', '160112_27P', '160123_08_LOOK']
['160126_08_LOOK', '160305_01p', '160305_09P', '160317_16_LOOK']
['160317_20P', '160318_16', '160318_21p', '151210_05P']
```


### Viewing morphology



```python
import golding_mso as gmso

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
mso_cell.topology()
```

```
516 lines read
|--------------|   soma[0]
                `------------|   dend[0]
                                `----------|   dend[2]
                                            `----------------------|   dend[6]
                                            `----|   dend[3]
                                                `----|   dend[5]
                                                `------|   dend[4]
                                `------------------------------------------|   dend[1]
                `------------------------------------|   apic[0]
                                                        `----------|   apic[4]
                                                                    `--|   apic[6]
                                                                    `----|   apic[5]
                                                        `--------|   apic[1]
                                                                `--------------|   apic[3]
                                                                `------------|   apic[2]
```
    




```python
import golding_mso as gmso
from neuron import h
import matplotlib.pyplot as plt

fig = plt.figure()
mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
ps = h.PlotShape(h.SectionList(mso_cell.allsec_nofilopodia))
ps.show(0)
ps.plot(fig)
```

```
516 lines read





<Axis3DWithNEURON: >
```

![Matplotlib morphology visual](/readme_files/demo_7_2.png)
    


### Changing cell parameters



```python
import golding_mso as gmso

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
mso_cell.assign_channels()
mso_cell.attach_axon()

print(f"Original resting potential: {mso_cell.resting_potential} mV\n")

print("Modifying KHT channel mechanism and KLT conductance...\n")
mso_cell.channels['kht']['mechanism'] = 'leak_klt' # Replace KHT mechanism with passive channel
mso_cell.conductances['soma']['klt'] *= 2  # Double the KLT conductance in soma

print(f"Updated resting potential: {mso_cell.resting_potential} mV")

```

```
516 lines read
Original resting potential: -60.99978370571063 mV

Modifying KHT channel mechanism and KLT conductance...

Updated resting potential: -65.39322834144977 mV
```

### Current injection



```python
import golding_mso as gmso
import matplotlib.pyplot as plt
from neuron import h

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
mso_cell.assign_channels()
mso_cell.attach_axon()

stim = h.IClamp(mso_cell.somatic[0](0.5))
stim.amp = 1.3
stim.dur = 10
stim.delay = 10

soma_v = h.Vector().record(mso_cell.somatic[0](0.5)._ref_v)
axon_v = h.Vector().record(mso_cell.nodes[-1](0.5)._ref_v)
t = h.Vector().record(h._ref_t)

h.finitialize(-58)
h.continuerun(10)
h.frecord_init()
h.continuerun(13)

plt.plot(t, soma_v, label='Soma', color='blue')
plt.plot(t, axon_v, label='Axon', color='orange')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('MSO Cell Response to Current Injection')
plt.legend()
plt.show()
```

```    
516 lines read
```



    
![Model current injection](/readme_files/demo_11_1.png)
    


### Propagation testing



```python
import golding_mso as gmso
import numpy as np
import matplotlib.pyplot as plt
from golding_mso.sims import propagation_test
from neuron import h

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
mso_cell.cvode.active(0)
mso_cell.assign_channels()
mso_cell.attach_axon()

prop_data = propagation_test(mso_cell, mso_cell.lateral_nofilopodia)
for entry, sec in zip(prop_data, mso_cell.lateral_nofilopodia):
    plt.plot([h.distance(seg, mso_cell.somatic[0](0.5)) for seg in sec],abs(np.subtract(entry['rec_site']['maxt'],entry['syn']['maxt'])), label=f'Section {sec.name()}')
plt.xlabel('Distance from Soma (µm)')
plt.ylabel('Propagation Delay (ms)')

```

```  
516 lines read





Text(0, 0.5, 'Propagation Delay (ms)')
```




    
![Propagation delay vs. distance plot](/readme_files/demo_13_2.png)
    


### ITD testing



```python
import golding_mso as gmso
import matplotlib.pyplot as plt
import numpy as np
from golding_mso.sims import itd_test_sweep

mso_cell = gmso.Cell(gmso.morphologies['151124_03'])
mso_cell.assign_channels()
mso_cell.attach_axon()

itd_results = itd_test_sweep(mso_cell, mso_cell.lateral_nofilopodia, mso_cell.medial_nofilopodia, 'total', itd_vals=np.arange(-0.5,0.51,0.01), record_axon=True, exc_fiber_gmax = 0.03)  # Test ITDs of -0.25 to 0.25 ms
spikes = itd_results['spike_counts']
for trial in range(19):
    spikes += itd_test_sweep(mso_cell, mso_cell.lateral_nofilopodia, mso_cell.medial_nofilopodia, 'total', itd_vals=np.arange(-0.5,0.51,0.01), record_axon=True, exc_fiber_gmax = 0.03)['spike_counts']  # Test ITDs of -0.25 to 0.25 ms

plt.plot(itd_results['itd_vals'], spikes/21)
plt.xlabel('ITD (ms)')
plt.ylabel('Spike prob.')
plt.title('ITD Tuning Curve (n=20)')
plt.show()
```


    
![ITD tuning curve](/readme_files/demo_15_0.png)
    
### Editing config

```python
from golding_mso import get_config, set_config, reset_config
from pprint import pprint

reset_config()
current_config = get_config()
print("--- Original Configuration ---")
pprint(current_config)
print("\n--- Modifying 'Ra' in 'initialization' to 150 ---\n")

current_config['initialization']['Ra'] = 150
set_config(current_config)
print("--- Updated Configuration ---")
pprint(get_config())

reset_config();
```

```
--- Original Configuration ---
{'channels': {'hcn': {'cond_label': 'gbar',
                        'ion': 'h',
                        'mechanism': 'khurana_hcn',
                        'reversal_potential': -35},
                'kht': {'cond_label': 'gbar',
                        'ion': 'k',
                        'mechanism': 'nabel_kht',
                        'reversal_potential': -90},
                'klt': {'cond_label': 'gbar',
                        'ion': 'k',
                        'mechanism': 'mathews_klt',
                        'reversal_potential': -90},
                'leak': {'cond_label': 'gbar',
                        'ion': None,
                        'mechanism': 'leak',
                        'reversal_potential': -70},
                'na': {'cond_label': 'gbar',
                        'ion': 'na',
                        'mechanism': 'scott_na',
                        'reversal_potential': 69}},
    'conductances': {'cais': {'hcn': 0.002,
                            'klt': 0.155,
                            'leak': 5e-05,
                            'na': 0.25},
                    'dendrite': {'hcn': 0.001,
                                'kht': 0.00055,
                                'klt': 0.02,
                                'leak': 5e-05,
                                'na': 0},
                    'internode': {'leak': 2e-05},
                    'node': {'klt': 0.155, 'leak': 0.005, 'na': 0.25},
                    'soma': {'hcn': 0.001,
                            'kht': 0.00055,
                            'klt': 0.04,
                            'leak': 5e-05,
                            'na': 0.03},
                    'tais': {'hcn': 0.002,
                            'klt': 0.155,
                            'leak': 5e-05,
                            'na': 0.25}},
    'general': {'logging_level': 'ERROR'},
    'initialization': {'Ra': 200,
                    'cm': 0.9,
                    'compartment_size': 2,
                    'filopodia_maximum_diameter': 0.5,
                    'filopodia_maximum_length': 15,
                    'internode_cm': 0.0111,
                    'stabilization_time': 100}}

--- Modifying 'Ra' in 'initialization' to 150 ---

--- Updated Configuration ---
{'channels': {'hcn': {'cond_label': 'gbar',
                        'ion': 'h',
                        'mechanism': 'khurana_hcn',
                        'reversal_potential': -35},
                'kht': {'cond_label': 'gbar',
                        'ion': 'k',
                        'mechanism': 'nabel_kht',
                        'reversal_potential': -90},
                'klt': {'cond_label': 'gbar',
                        'ion': 'k',
                        'mechanism': 'mathews_klt',
                        'reversal_potential': -90},
                'leak': {'cond_label': 'gbar',
                        'ion': None,
                        'mechanism': 'leak',
                        'reversal_potential': -70},
                'na': {'cond_label': 'gbar',
                        'ion': 'na',
                        'mechanism': 'scott_na',
                        'reversal_potential': 69}},
    'conductances': {'cais': {'hcn': 0.002,
                            'klt': 0.155,
                            'leak': 5e-05,
                            'na': 0.25},
                    'dendrite': {'hcn': 0.001,
                                'kht': 0.00055,
                                'klt': 0.02,
                                'leak': 5e-05,
                                'na': 0},
                    'internode': {'leak': 2e-05},
                    'node': {'klt': 0.155, 'leak': 0.005, 'na': 0.25},
                    'soma': {'hcn': 0.001,
                            'kht': 0.00055,
                            'klt': 0.04,
                            'leak': 5e-05,
                            'na': 0.03},
                    'tais': {'hcn': 0.002,
                            'klt': 0.155,
                            'leak': 5e-05,
                            'na': 0.25}},
    'general': {'logging_level': 'ERROR'},
    'initialization': {'Ra': 150,
                    'cm': 0.9,
                    'compartment_size': 2,
                    'filopodia_maximum_diameter': 0.5,
                    'filopodia_maximum_length': 15,
                    'internode_cm': 0.0111,
                    'stabilization_time': 100}}
```

</details>
