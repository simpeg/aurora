#!/usr/bin/env python
# coding: utf-8

# ## Process Earthscope Data with Aurora

# This notebook shows how to process MTH5 data from the earthscope dataset.

# Steps
# 1. Define your mth5
# 2. Get a Run Summary from the mth5
# 3. Select the station to process and optionally the remote reference station
# 4. Create a processing config
# 5. Generate TFs
# 6. Archive the TFs (in emtf_xml or z-file)

# ### Here are the modules we will need to import 

# In[1]:


import pathlib
import warnings

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset

warnings.filterwarnings('ignore')


# ## Define mth5 file
# 
# The synthetic mth5 file is used for testing in `aurora/tests/synthetic/` and probably already exists on your system

# In[3]:


#from aurora.test_utils.earthscope.helpers import DATA_PATH
#print(DATA_PATH)
DATA_PATH = pathlib.Path("/home/kkappler/.cache/earthscope/data/")


# In[4]:


mth5_path = DATA_PATH.joinpath("8P_AZT14.h5")
print(mth5_path.exists())


# ## Get a Run Summary
# 
# Note that we didn't need to explicitly open the mth5 to do that, we can pass the path if we want

# In[5]:


mth5_run_summary = RunSummary()
mth5_run_summary.from_mth5s([mth5_path,])
run_summary = mth5_run_summary.clone()
run_summary.mini_summary


# ## Define a Kernel Dataset
# 

# In[6]:


kernel_dataset = KernelDataset()
# kernel_dataset.from_run_summary(run_summary, "test1", "test2")
kernel_dataset.from_run_summary(run_summary, "AZT14")
kernel_dataset.mini_summary


# ## Now define the processing Configuration
# 
# The only things we need to provide are our band processing scheme, and the data sample rate to generate a default processing configuration.
# 
# The config is then told about the stations via the kernel dataset.
# 
# **When doing only single station processing you need to specify RME processing (rather than remote reference processing which expects extra time series from another station)

# In[7]:


cc = ConfigCreator()
config = cc.create_from_kernel_dataset(kernel_dataset)


# ## Call process_mth5

# In[8]:


show_plot = True
tf_cls = process_mth5(config,
                    kernel_dataset,
                    units="MT",
                    show_plot=show_plot,
                    z_file_path=None,
                )


# In[ ]:


xml_file_base = f"synthetic_test1.xml"
tf_cls.write_tf_file(fn=xml_file_base, file_type="emtfxml")


# In[ ]:




