from ouster import client
from ouster import pcap
from os.path import join
import open3d as o3d

def get_source_and_metadata(parent, name, metadata=None):
    if metadata is None:
        metadata = name
    metadata = join(parent,f'{metadata}.json')
    pcap_ = join(parent,f'{name}.pcap')
    print("Loading metadata: ", metadata)
    print(pcap_)
    with open(metadata, 'r') as f:
        metadata = client.SensorInfo(f.read())

    source = pcap.Pcap(pcap_, metadata)

    return source, metadata
    

def get_pcds_at_frame(frames=[0]):

    parent = join ('..','data', 'Boston_OS1-128_2023-01-02', 'Back_1_Subaru', 'split_files')
    # parent = join('..', 'data','USDOT_SMART')

    file_names = [
        '2'
        # 'record-2022-11-13T205426.416Z'
    # ,  'record-2022-11-13T205435.923Z'
    #  'record-2022-11-13T213535.201Z'
    #  'record-2022-11-13T213543.065Z'
    ]

    print("Started loading frames")

    frames = set(frames)

    pcds = []

    for file_name in file_names:
        pcap, metadata = get_source_and_metadata(parent, file_name,metadata="meta")

        xyzlut = client.XYZLut(metadata)
        scans = iter(client.Scans(pcap))
        print("\tLoading file: ", file_name)
        x = 1
        for i in range(max(frames)):
            if i+1 in frames:
                scan = next(scans)    

                range_ = scan.field(client.ChanField.RANGE)
                xyz = xyzlut(range_)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1,3)))

                pcds.append(pcd)
                x+=1
            
            if x % 10 == 0:
                print("\t\tLoading frames: ", i+1)

    return pcds