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

#ifndef PCIE_TOPO_HPP
#define PCIE_TOPO_HPP
#ifndef LIB_IBVERBS_UNAVAILABLE
#include <iostream>
#include "Compatibility.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <set>

static std::vector<std::string> IbDeviceBusIds;
static std::vector<std::set<int>> NicToGpuMapper;
static std::vector<int> GpuToNicMapper;
static std::vector<std::string> DeviceNames;
static int DeviceCount;
static bool Initialized = false;
#define INIT_ONCE(ret)  \
  do {                  \
  if(Initialized)       \
  {                     \
    return ret;         \
  }                     \
  Initialized = true;   \
  } while(0);

class PCIe_tree 
{
public:
  std::set<PCIe_tree> children;
  std::string address;
  std::string description;

  // Constructor
  PCIe_tree(const std::string& addr) : address(addr) {}
  
    // Constructor
  PCIe_tree(const std::string& addr, const std::string& desc)
           :address(addr), description(desc) {}

  // Default constructor
  PCIe_tree() : address(""), description("") {}

  // Comparison operator for std::set
  bool operator<(const PCIe_tree& other) const {
    return address < other.address;
  }

  // Function to find a node by address
  const PCIe_tree* find(const std::string& addr) const {
    if (address == addr) {
      return this;
    }
    for (const auto& child : children) {
      const PCIe_tree* result = child.find(addr);
      if (result) {
        return result;
      }
    }
    return nullptr;
  }
};

static PCIe_tree pcie_root;

static void insert_pcie_path_to_tree(PCIe_tree* root, const std::string& pcieAddress, const std::string& description)
{
  std::filesystem::path devicePath = "/sys/bus/pci/devices/" + pcieAddress;
  if (!std::filesystem::exists(devicePath))
  {
    printf("[ERROR] Device path %s does not exist\n", devicePath.c_str());
    return;
  }
  std::string canonicalPath = std::filesystem::canonical(devicePath).string();
  std::istringstream iss(canonicalPath);
  std::string token;
  PCIe_tree* currentNode = root;

  bool ignore = true;
  while (std::getline(iss, token, '/'))
  {
    std::string address = token;
    auto it = currentNode->children.find(PCIe_tree(address));
    if (it == currentNode->children.end())
    {
      currentNode->children.insert(PCIe_tree(address));
      it = currentNode->children.find(PCIe_tree(address));
    }
    currentNode = const_cast<PCIe_tree*>(&(*it));
  }
  currentNode->description = description;
}

static const PCIe_tree* find_lca_between_two_nodes(const PCIe_tree* root, std::string node1, std::string node2)
{
  if (!root || root->address == node1 || root->address == node2)
  {
    return root;
  }

  const PCIe_tree* leftLCA = nullptr;
  const PCIe_tree* rightLCA = nullptr;

  for (const auto& child : root->children)
  {
    const PCIe_tree* lca = find_lca_between_two_nodes(&child, node1, node2);
    if (lca)
    {
      if (leftLCA)
      {
        rightLCA = lca;
        break;
      } 
      else
      {
        leftLCA = lca;
      }
    }
  }

  if (leftLCA && rightLCA) {
    return root;
  }

  return leftLCA ? leftLCA : rightLCA;
}

static int get_lca_depth(const std::string targetBusID, const PCIe_tree* node, int depth = 0)
{
  if (!node)
  {
    return -1;
  }
  if (targetBusID == node->address)
  {
    return depth;
  }
  for (const auto& child : node->children)
  {
    int distance = get_lca_depth(targetBusID, &child, depth + 1);
    if (distance != -1)
    {
      return distance;
    }
  }
  return -1;
}

// Function to extract the bus number from a PCIe address (domain:bus:device.function)
static int extract_bus_number(const std::string& pcieAddress)
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

// Function to compute the distance between two bus IDs 
static int get_bus_id_distance(const std::string& pcieAddress1, const std::string& pcieAddress2)
{
  int bus1 = extract_bus_number(pcieAddress1);
  int bus2 = extract_bus_number(pcieAddress2);

  if (bus1 == -1 || bus2 == -1)
  {
    return -1; // Error case, invalid bus number
  }

  // Distance between two PCIe devices based on their bus numbers
  return std::abs(bus1 - bus2);
}

static int get_nearest_pcie_device_in_tree(const PCIe_tree& root, const std::string busID, const std::vector<std::string>& targetBusIds)
{
  int max_depth = -1;
  int index_of_closest = -1;
  std::vector <int> matches;
  for (const auto& targetBusID : targetBusIds)
  {
    if (targetBusID.empty()) continue;    
    const PCIe_tree* lca = find_lca_between_two_nodes(&root, busID, targetBusID);
    if (lca)
    {
      int depth = get_lca_depth(lca->address, &pcie_root);
      if (depth > max_depth)
      {        
        max_depth = depth;
        index_of_closest = &targetBusID - &targetBusIds[0];
        matches.clear(); // found a new max depth
        matches.push_back(index_of_closest);
      }
      else if(depth == max_depth && depth >= 0)
      {
        matches.push_back(&targetBusID - &targetBusIds[0]);
      }
    }
  }
  // when more than one LCA match is found, opt for the one with the smallest
  // bus id difference
  if(matches.size() > 1)
  {
    int minDistance = std::numeric_limits<int>::max();
    for (const auto& match : matches)
    {
      int distance = get_bus_id_distance(busID, targetBusIds[match]);
      if (distance < minDistance)
      {
        minDistance = distance;
        index_of_closest = match;
      }
    }
  }
  return index_of_closest;
}

static void init_device_paths_and_build_pcie_tree()
{
  struct ibv_device **dev_list;
  dev_list = ibv_get_device_list(&DeviceCount);
  if (!dev_list)
  {
    std::cerr << "Failed to get IB devices list." << std::endl;
    return;
  }
  IbDeviceBusIds.resize(DeviceCount, "");
  NicToGpuMapper.resize(DeviceCount);
  DeviceNames.resize(DeviceCount);
  int closestDevice = -1;
  int minDistance = std::numeric_limits<int>::max();

  for (int i = 0; i < DeviceCount; ++i)
  {
    struct ibv_device *device = dev_list[i];
    DeviceNames[i] = device->name;
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
        insert_pcie_path_to_tree(&pcie_root, nicBusId, DeviceNames[i]);
      }
    }
  }
  ibv_free_device_list(dev_list);  
  int numHipDevices;
  HIP_CALL(hipGetDeviceCount(&numHipDevices));
  for (int i = 0; i < numHipDevices; ++i)
  {
    char hipPciBusId[64];
    hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i);
    if (err != hipSuccess) 
    {
      std::cerr << "Failed to get PCI Bus ID for HIP device " << i << ": " << hipGetErrorString(err) << std::endl;   
      return;   
    }
    insert_pcie_path_to_tree(&pcie_root, hipPciBusId, "GPU " + std::to_string(i));
  }  
}

static int get_closest_rdma_nic_id(int hipDeviceId, bool useTopoTree = true)
{   
  char hipPciBusId[64];
  hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), hipDeviceId);
  if (err != hipSuccess) 
  {
    std::cerr << "Failed to get PCI Bus ID for HIP device " << hipDeviceId << ": " << hipGetErrorString(err) << std::endl;
    return -1;
  }
  int closestRdmaNicId = get_nearest_pcie_device_in_tree(pcie_root, hipPciBusId, IbDeviceBusIds);
  // The following will only use distance between bus IDs 
  // to determine the closest NIC to GPU if the PCIe tree approach fails
  if(closestRdmaNicId < 0)
  {
    printf("[Warn] falling back to PCIe bus ID distance to determine proximity\n");
    int minDistance = std::numeric_limits<int>::max();
    for (int i = 0; i < IbDeviceBusIds.size(); ++i)
    { 
      auto address = IbDeviceBusIds[i];
      if (address != "") {
        int distance = get_bus_id_distance(hipPciBusId, address);
        if (distance < minDistance && distance >= 0)
        {
          minDistance = distance;
          closestRdmaNicId = i;
        }
      }
    }  
  }
  return closestRdmaNicId;    
}

static int get_closest_gpu_device_id(int IbvDeviceId)
{
  init_device_paths_and_build_pcie_tree();
  assert(IbvDeviceId < IbDeviceBusIds.size());
  auto address = IbDeviceBusIds[IbvDeviceId];
  if (address == "") return -1;
  int numHipDevices;
  HIP_CALL(hipGetDeviceCount(&numHipDevices));
  GpuToNicMapper.resize(numHipDevices, -1);
  int closestDevice = -1;
  int minDistance = std::numeric_limits<int>::max();
  for (int i = 0; i < numHipDevices; ++i)
  {
    char hipPciBusId[64];
    hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i);
    if (err != hipSuccess) 
    {
      std::cerr << "Failed to get PCI Bus ID for HIP device " << i << ": " << hipGetErrorString(err) << std::endl;
      return -1;
    }
    int distance = get_bus_id_distance(hipPciBusId, address);
    if (distance < minDistance && distance >= 0)
    {
      minDistance = distance;
      closestDevice = i;
    }
  }
  return closestDevice;
}

static void init_device_mappings()
{
  INIT_ONCE();
  int numHipDevices;
  init_device_paths_and_build_pcie_tree();
  HIP_CALL(hipGetDeviceCount(&numHipDevices));
  GpuToNicMapper.resize(numHipDevices, -1);
  const char* closestNicEnv = std::getenv("CLOSEST_NIC");
  if (closestNicEnv)
  {
    std::istringstream iss(closestNicEnv);
    std::string token;
    int i = 0; 
    while (std::getline(iss, token, ','))
    {
      try
      {
        int nicId = std::stoi(token);
        if (nicId >= 0 && nicId < DeviceCount)
        {
          GpuToNicMapper[i] = nicId;
          assert(nicId < NicToGpuMapper.size());
          NicToGpuMapper[nicId].insert(i);
          i++;
        }
        else
        {
          std::cerr << "[Error] Invalid NIC ID in CLOSEST_NIC environment variable: " << nicId << std::endl;
          exit(1);
        }        
      }
      catch (const std::invalid_argument& e)
      {
        std::cerr << "[Error] Invalid NIC ID in CLOSEST_NIC environment variable: " << token << std::endl;
        exit(1);
      }
    }
    if(i < numHipDevices)
    {
      std::cerr << "[Error] Number of entries in CLOSEST_NIC environment variable is less than the number of detected GPUs: " << numHipDevices<< std::endl;
      exit(1);
    }
  }
  else 
  {
    for (int i = 0; i < numHipDevices; ++i)
    {
      int closestIbDevice = get_closest_rdma_nic_id(i);
      GpuToNicMapper[i] = closestIbDevice;
      if(closestIbDevice >= 0)
      {
        assert(closestIbDevice < NicToGpuMapper.size());
        NicToGpuMapper[closestIbDevice].insert(i);
      }
    }
  }  
}

int GetClosestIbDevice(int hipDeviceId)
{
  init_device_mappings();
  assert(hipDeviceId < GpuToNicMapper.size());
  return GpuToNicMapper[hipDeviceId];
}
void PrintPCIeTree(const PCIe_tree& node, const std::string& prefix = "", bool isLast = true)
{
  if(!node.address.empty())
  {
    std::cout << prefix << (isLast ? "└── " : "├── ") << node.address;
    if(!node.description.empty())
    {
      std::cout << "(" << node.description << ")";
    }
    std::cout<< std::endl;
  }
  const auto& children = node.children;
  for (auto it = children.begin(); it != children.end(); ++it)
  {
    PrintPCIeTree(*it, prefix + (isLast ? "    " : "│   "), std::next(it) == children.end());
  }
}

void PrintNicToGPUTopo(bool printAsCsv)
{
  init_device_mappings();
  if (printAsCsv)
  {
    std::cout << "Device Index,Device Name,Port Active,Closest GPU(s)" << std::endl;
  }
  else
  {
    std::cout << "Device Index | Device Name | Port Active | Closest GPU(s)| PCIe Bus ID" << std::endl;
    std::cout << "-------------+-------------+-------------+---------------+------------" << std::endl;
  }

  for (int i = 0; i < IbDeviceBusIds.size(); ++i)
  {
    std::string nicDevice = DeviceNames[i];
    bool portActive = IbDeviceBusIds[i] != "";
    std::string closestGpus;
    for (auto it = NicToGpuMapper[i].begin(); it != NicToGpuMapper[i].end(); ++it)
    {
      closestGpus += std::to_string(*it);
      if (std::next(it) != NicToGpuMapper[i].end())
      {
        closestGpus += ",";
      }
    }
    if (printAsCsv)
    {
      std::cout << i << ","
          << nicDevice << "," 
          << (portActive ? "Yes" : "No") << ","
          << closestGpus <<  ","
          << IbDeviceBusIds[i] <<std::endl;
    }
    else
    {
      std::cout << std::left << std::setw(12) << i << " | "
          << std::left << std::setw(11) << nicDevice << " | "
          << std::left << std::setw(11) << (portActive ? "Yes" : "No") << " | "
          << std::left << std::setw(13) << closestGpus << " | "
          << std::left << std::setw(11) << IbDeviceBusIds[i] 
          << std::endl;
    }
  }
  std::cout << std::endl;
  if (std::getenv("SHOW_TOPO_TREE"))
  {
    std::cout << "--------------------------" << std::endl;
    std::cout << "PCIe Tree (NICs and GPUs):" << std::endl;
    std::cout << "--------------------------" << std::endl;
    PrintPCIeTree(pcie_root);
    std::cout << std::endl;
  }

}

#else
int GetClosestIbDevice(int hipDeviceId)
{
  return -1;
}
void PrintNicToGPUTopo(bool printAsCsv) { }
#endif
#endif // PCIE_TOPO_HPP