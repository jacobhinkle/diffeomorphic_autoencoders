#!/bin/bash
#
#================================================================
# download_oasis_freesurfer.sh
#================================================================
#
# Usage: ./download_oasis_freesurfer.sh <input_file.csv> <directory_name> <xnat_central_username>
# 
# Download Freesurfer files from OASIS3 on XNAT Central and organize the files
#
# Required inputs:
# <input_file.csv> - A Unix formatted, comma-separated file containing a column for freesurfer_id 
#       (e.g. OAS30001_Freesurfer53_d0129)
# <directory_name> - A directory path (relative or absolute) to save the Freesurfer files to
# <xnat_central_username> - Your XNAT Central username used for accessing OASIS data on central.xnat.org
#       (you will be prompted for your password before downloading)
#
# This script organizes the files into folders like this:
#
# directory_name/OAS30001_MR_d0129/$FREESURFER_FOLDERS
#
#
# Last Updated: 9/6/2018
# Author: Sarah Keefe
#
#
unset module

# usage instructions
if [ ${#@} == 0 ]; then
    echo ""
    echo "OASIS Freesurfer download script"
    echo ""
    echo "This script downloads Freesurfer files based on a list of session ids in a csv file. "
    echo ""   
    echo "Usage: $0 input_file.csv directory_name central_username scan_type"
    echo "<input_file>: A Unix formatted, comma separated file containing the following columns:"
    echo "    freesurfer_id (e.g. OAS30001_Freesurfer53_d0129)"
    echo "<directory_name>: Directory path to save Freesurfer files to"  
    echo "<xnat_central_username>: Your XNAT Central username used for accessing OASIS data (you will be prompted for your password)"  
else 

    # Get the input arguments
    INFILE=$1
    DIRNAME=$2
    USERNAME=$3

    # Create the directory if it doesn't exist yet
    if [ ! -d $DIRNAME ]
    then
        mkdir $DIRNAME
    fi

    # Read in password
    read -s -p "Enter your password for accessing OASIS data on XNAT Central:" PASSWORD

    echo ""

    # Get a JSESSION for authentication to XNAT
    JSESSION=`curl -k -s -u $USERNAME:$PASSWORD "https://central.xnat.org/REST/JSESSION"` # get a session to authenticate with
    #JSESSION=BA16CA2317220ED9EA74B99AD6178D09
    if [[ ${#JSESSION} -gt 100 ]]
    then
        echo JSESSION request returned non-session ID: $JSESSION
        exit 1
    fi
    echo JSESSION=$JSESSION

    # Read the file
    sed 1d $INFILE | while IFS=, read -r FREESURFER_ID; do

        # Get the subject ID from the first part of the experiment ID (OAS30001 from ID OAS30001_Freesurfer53_d0129)
        SUBJECT_ID=`echo $FREESURFER_ID | cut -d_ -f1`

        # Get the days from entry from the third part of the experiment ID (d0129 from ID OAS30001_Freesurfer53_d0129)
        DAYS_FROM_ENTRY=`echo $FREESURFER_ID | cut -d_ -f3`

        # combine to form the experiment label (OAS30001_MR_d0129)
        EXPERIMENT_LABEL=${SUBJECT_ID}_MR_${DAYS_FROM_ENTRY}

        #echo "Checking for Freesurfer ID ${FREESURFER_ID} associated with ${EXPERIMENT_LABEL}."

        filename=$DIRNAME/${FREESURFER_ID}_skullstripped.mgz
        if [[ ( ! -f $filename ) || ( -f ${filename}_FAILED ) ]]
        then
            touch ${filename}_FAILED 
            FILELIST=$(curl -s -k -b JSESSIONID=$JSESSION https://central.xnat.org/data/archive/projects/OASIS3/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_LABEL}/assessors/${FREESURFER_ID}/files?format=csv)
            if [[ "$?" -ne "0" ]]
            then
                echo File list retrieval returned code $? with message: $FILELIST
            fi
            BM_PATH=$(echo "$FILELIST" | grep brainmask.mgz | grep -v trash | head -n1 | awk -F, '{print $3}')
            if [ "$BM_PATH" == "" ]
            then
                echo Could not find brainmask.mgz in file list:
                echo $FILELIST
                continue
            fi
            download_url=https://central.xnat.org$BM_PATH
            echo $download_url
            echo -n "Downloading $filename ... "
            curl -s -k -b JSESSIONID=$JSESSION -o $filename $download_url
            CURL_RET=$?
            # check that the download is a valid gzipped file
            IS_HTML=$(file $filename | grep HTML | wc -l)
            if [ "$IS_HTML" -ne "0" ]
            then
                # do further check with gzip, just to be sure
                echo Download of $FREESURFER_ID resulted in HTML:
                cat $filename
                exit 1
            elif [ "$CURL_RET" -eq "0" ]
            then
                echo "Checks passed"
                rm ${filename}_FAILED
            else
                echo "Download failed with code $CURL_RET"
            fi
        fi


        continue
        # Check the zip file to make sure we downloaded something
        # If the zip file is invalid, we didn't download a scan so there is probably no scan of that type
        # If the zip file is valid, unzip and rearrange the files
        if zip -Tq $DIRNAME/$FREESURFER_ID.zip > /dev/null; then
            # We found a successfully downloaded valid zip file

            echo "Downloaded a Freesurfer (${FREESURFER_ID}) from ${EXPERIMENT_LABEL}." 

            echo "Unzipping Freesurfer and rearranging files."

            # Unzip the downloaded file
            unzip $DIRNAME/$FREESURFER_ID.zip -d $DIRNAME

            # Rearrange the files so there are fewer subfolders
            # Move the main Freesurfer subfolder up 5 levels
            # Ends up like this:
            # directory_name/OAS30001_MR_d0129/freesurfer_folders
            # directory_name/OAS30001_MR_d0129/etc
            mv $DIRNAME/$FREESURFER_ID/out/resources/DATA/files/* $DIRNAME/.

            # do this so we don't have to use rm -rf. 
            rmdir $DIRNAME/$FREESURFER_ID/out/resources/DATA/files
            rmdir $DIRNAME/$FREESURFER_ID/out/resources/DATA
            rmdir $DIRNAME/$FREESURFER_ID/out/resources
            rmdir $DIRNAME/$FREESURFER_ID/out
            rmdir $DIRNAME/$FREESURFER_ID

            # Remove the Freesurfer zip file that the files were moved from
            rm -r $DIRNAME/$FREESURFER_ID.zip
        else
            echo "Could not get Freesurfer ${FREESURFER_ID} in ${EXPERIMENT_LABEL}."           
        fi

        echo "Done with ${FREESURFER_ID}."

    done < $INFILE
fi
