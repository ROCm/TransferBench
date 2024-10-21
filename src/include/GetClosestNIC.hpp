/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef GET_CLOSEST_NIC_HPP
#define GET_CLOSEST_NIC_HPP
#ifndef LIB_IBVERBS_UNAVAILABLE
#include <iostream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <infiniband/verbs.h>

static std::vector<std::string> IbDeviceBusIds;
static bool Initialized = false;

// Function to extract the bus number from a PCIe address (domain:bus:device.function)
int GetBusNumber(const std::string& pcieAddress)
{
  int domain, bus, device, function;
  char delimiter;

  std::istringstream iss(pcieAddress);
  iss >> std::hex >> domain >> delimiter >> bus >> delimiter >> device >> delimiter >> function;

  if (iss.fail())
  {
    std::cerr << "Invalid PCIe address format: " << pcieAddress << std::endl;
    return -1; // Invalid bus number
  }

  return bus;
}

// Function to compute the distance between two PCIe addresses
int GetPcieDistance(const std::string& pcieAddress1, const std::string& pcieAddress2)
{
  int bus1 = GetBusNumber(pcieAddress1);
  int bus2 = GetBusNumber(pcieAddress2);

  if (bus1 == -1 || bus2 == -1)
  {
    return -1; // Error case, invalid bus number
  }

  // Distance between two PCIe devices based on their bus numbers
  return std::abs(bus1 - bus2);
}

static void InitIbDevicePaths()
{
  if(Initialized)
  {
    return;
  }
  Initialized = true;
  struct ibv_device **dev_list;
  int num_devices;
  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list)
  {
    std::cerr << "Failed to get IB devices list." << std::endl;
    return;
  }
  IbDeviceBusIds.resize(num_devices, "");
  int closestDevice = -1;
  int minDistance = std::numeric_limits<int>::max();

  for (int i = 0; i < num_devices; ++i)
  {
    struct ibv_device *device = dev_list[i];
    struct ibv_context *context = ibv_open_device(device);
    if (!context)
    {
      std::cerr << "Failed to open device " << device->name << std::endl;
      continue;
    }

    struct ibv_device_attr device_attr;
    if (ibv_query_device(context, &device_attr))
    {
      std::cerr << "Failed to query device attributes for " << device->name << std::endl;
      ibv_close_device(context);
      continue;
    }

    bool portActive = false;
    for (int port = 1; port <= device_attr.phys_port_cnt; ++port)
    {
      struct ibv_port_attr port_attr;
      if (ibv_query_port(context, port, &port_attr))
      {
        std::cerr << "Failed to query port " << port << " attributes for " << device->name << std::endl;
        continue;
      }
      if (port_attr.state == IBV_PORT_ACTIVE)
      {
        portActive = true;
        break;
      }
    }

    ibv_close_device(context);

    if (!portActive)
    {        
      continue;
    }

    std::string device_path(device->dev_path);
    if (std::filesystem::exists(device_path))
    {
      std::string pciPath = std::filesystem::canonical(device_path + "/device").string();
      std::size_t pos = pciPath.find_last_of('/');
      if (pos != std::string::npos) {
        std::string nicBusId = pciPath.substr(pos + 1);
        IbDeviceBusIds[i] = nicBusId;        
      }
    }
  }

  ibv_free_device_list(dev_list);  
}

int GetClosestIbDevice(int hipDeviceId)
{
  InitIbDevicePaths();
  char hipPciBusId[64];
  hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), hipDeviceId);
  if (err != hipSuccess) 
  {
    std::cerr << "Failed to get PCI Bus ID for HIP device " << hipDeviceId << ": " << hipGetErrorString(err) << std::endl;
    return -1;
  }

  int closestDevice = -1;
  int minDistance = std::numeric_limits<int>::max();

  for (int i = 0; i < IbDeviceBusIds.size(); ++i)
  { 
    auto address = IbDeviceBusIds[i];
    if (address != "") {
      int distance = GetPcieDistance(hipPciBusId, address);
      if (distance < minDistance && distance >= 0)
      {
        minDistance = distance;
        closestDevice = i;
      }
    }
  }
  return closestDevice;
}
#else
int GetClosestIbDevice(int hipDeviceId)
{
  return -1;
}
#endif
#endif // GET_CLOSEST_NIC_HPP