# forecapp-api

## Minimum requirements

- 1 vCPU
- 2 GB RAM
- 15 GB Disk

## Setup instructions

    eb init

    Select a default region
    1) us-east-1 : US East (N. Virginia)
    2) us-west-1 : US West (N. California)
    3) us-west-2 : US West (Oregon)
    4) eu-west-1 : EU (Ireland)
    5) eu-central-1 : EU (Frankfurt)
    6) ap-south-1 : Asia Pacific (Mumbai)
    7) ap-southeast-1 : Asia Pacific (Singapore)
    8) ap-southeast-2 : Asia Pacific (Sydney)
    9) ap-northeast-1 : Asia Pacific (Tokyo)
    10) ap-northeast-2 : Asia Pacific (Seoul)
    11) sa-east-1 : South America (Sao Paulo)
    12) cn-north-1 : China (Beijing)
    13) cn-northwest-1 : China (Ningxia)
    14) us-east-2 : US East (Ohio)
    15) ca-central-1 : Canada (Central)
    16) eu-west-2 : EU (London)
    17) eu-west-3 : EU (Paris)
    18) eu-north-1 : EU (Stockholm)
    19) eu-south-1 : EU (Milano)
    20) ap-east-1 : Asia Pacific (Hong Kong)
    21) me-south-1 : Middle East (Bahrain)
    22) af-south-1 : Africa (Cape Town)
    (default is 3): 2


    Select an application to use
    1) forecapp-api
    2) [ Create new Application ]
    (default is 1):


    It appears you are using Docker. Is this correct?
    (Y/n): n
    Select a platform.
    1) .NET Core on Linux
    2) .NET on Windows Server
    3) Docker
    4) Go
    5) Java
    6) Node.js
    7) PHP
    8) Packer
    9) Python
    10) Ruby
    11) Tomcat
    (make a selection): 9

    Select a platform branch.
    1) Python 3.8 running on 64bit Amazon Linux 2
    2) Python 3.7 running on 64bit Amazon Linux 2
    (default is 1):

    Do you wish to continue with CodeCommit? (Y/n): n
    Do you want to set up SSH for your instances?
    (Y/n): y

    Select a keypair.
    1) aws-forecapp
    2) [ Create new KeyPair ]
    (default is 1): 1


    eb create -i t3.medium --single

    Enter Environment Name
    (default is forecapp-api-dev): forecapp-api-env
    Enter DNS CNAME prefix
    (default is forecapp-api-env): forecapp-api
