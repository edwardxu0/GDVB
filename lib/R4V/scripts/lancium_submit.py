#!/usr/bin/env python3
"""
Script to submit jobs to the Lancium job queue.

Author: David Shriver
Date: 2019-09-23
"""
import argparse
import os
import shlex
import subprocess as sp
import tempfile
import xml.etree.ElementTree as ET

from pathlib import Path
from xml.dom import minidom


def memory_t(value):
    unit = value[-1].lower()
    if unit in [str(i) for i in range(10)]:
        return value
    value = int(value[:-1])
    if unit == "k":
        return str(value * 1000)
    if unit == "m":
        return str(value * 1000000)
    if unit == "g":
        return str(value * 1000000000)
    raise argparse.ArgumentTypeError("Unknown memory unit: %s" % unit)


def time_t(value):
    try:
        return float(value) * 60
    except:
        pass
    days_time = value.split("-")
    if len(days_time) > 1:
        days, time = days_time
    else:
        days = 0
        time = days_time[0]
    hms = time.split(":")
    hours, minutes, seconds = 0, 0, 0
    if len(hms) == 3:
        hours, minutes, seconds = hms
    elif len(hms) == 2 and len(days_time) == 1:
        minutes, seconds = hms
    elif len(hms) == 2 and len(days_time) == 2:
        hours, minutes = hms
    elif len(hms) == 1 and len(days_time) == 2:
        hours = hms[0]
    else:
        raise argparse.ArgumentTypeError("Unsupported time format: %s" % value)
    return (
        (float(days) * 24 * 60 * 60)
        + (float(hours) * 60 * 60)
        + (float(minutes) * 60)
        + (float(seconds))
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--grid-path", required=True, type=Path)
    parser.add_argument(
        "-I",
        "--input-data",
        action="append",
        default=[],
        type=Path,
        help="Input files to stage.",
    )
    parser.add_argument(
        "-O",
        "--output_data",
        action="append",
        default=[],
        type=Path,
        help="Output files to stage.",
    )
    parser.add_argument(
        "-J", "--job-name", default="runJob", type=str, help="A name for the job."
    )
    parser.add_argument(
        "-e", "--error", default=Path("std.err"), type=Path, help="Path to save stderr."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("std.out"),
        type=Path,
        help="Path to save stdout.",
    )
    parser.add_argument("-t", "--time", type=time_t, default=None, help="Time limit.")
    parser.add_argument(
        "--mem", type=memory_t, default=None, help="Memory requirement."
    )
    parser.add_argument(
        "--gres", type=str, default=None, help="Generic resources required."
    )
    parser.add_argument("executable", type=str, help="Script to run.")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments for the script."
    )
    return parser.parse_args()


def job_definition():
    el = ET.Element("jsdl:JobDefinition")
    el.set("xmlns:jsdl-hpcpa", "http://schemas.ggf.org/jsdl/2006/07/jsdl-hpcpa")
    el.set("xmlns:hpcfse-ac", "http://schemas.ogf.org/hpcp/2007/11/ac")
    el.set("xmlns:jsdl-sweep", "http://schemas.ogf.org/jsdl/2009/03/sweep")
    el.set("xmlns:jsdl-posix", "http://schemas.ggf.org/jsdl/2005/11/jsdl-posix")
    el.set("xmlns:jsdl-spmd", "http://schemas.ogf.org/jsdl/2007/02/jsdl-spmd")
    el.set("xmlns:jsdl", "http://schemas.ggf.org/jsdl/2005/11/jsdl")
    el.set(
        "xmlns:wss",
        "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd",
    )
    el.set("xmlns:ns8", "http://vcgr.cs.virginia.edu/jsdl/genii")
    el.set("xmlns:ns9", "http://schemas.ogf.org/jsdl/2009/03/sweep/functions")
    return el


def job_description(parent):
    el = ET.SubElement(parent, "jsdl:JobDescription")
    return el


def job_identification(parent, job_name):
    el = ET.SubElement(parent, "jsdl:JobIdentification")
    name = ET.SubElement(el, "jsdl:JobName")
    name.text = job_name
    return el


def job_application(parent, executable, args, output, error):
    el = ET.SubElement(parent, "jsdl:Application")
    posix_app = ET.SubElement(el, "jsdl-posix:POSIXApplication")
    exe = ET.SubElement(posix_app, "jsdl-posix:Executable")
    exe.text = executable
    for arg in args:
        a = ET.SubElement(posix_app, "jsdl-posix:Argument")
        a.text = arg
    out = ET.SubElement(posix_app, "jsdl-posix:Output")
    out.text = output.name
    err = ET.SubElement(posix_app, "jsdl-posix:Error")
    err.text = error.name
    return el


def job_resources(parent):
    el = ET.SubElement(parent, "jsdl:Resources")
    return el


def job_time(parent, time):
    if time is not None:
        t = ET.SubElement(parent, "ns8:WallclockTime")
        t_ub = ET.SubElement(t, "jsdl:UpperBoundedRange", exclusiveBound="false")
        t_ub.text = str(time)
    return parent


def job_memory(parent, memory):
    if memory is not None:
        mem = ET.SubElement(parent, "jsdl:TotalPhysicalMemory")
        mem_ub = ET.SubElement(mem, "jsdl:UpperBoundedRange", exclusiveBound="false")
        mem_ub.text = memory
    return parent


def job_gres(parent, gres):
    if gres is not None:
        for req in gres.split(","):
            resource, *type_count = gres.split(":")
            if resource.lower() == "gpu":
                gpu = ET.SubElement(
                    parent, "ns8:property", name="supports:gpu", value="true"
                )
                gpu_count = ET.SubElement(parent, "jsdl:GPUCountPerNode")
                gpu_count_ub = ET.SubElement(
                    gpu_count, "jsdl:UpperBoundedRange", exclusiveBound="false"
                )
                gpu_count_ub.text = type_count[-1]
                if len(type_count) == 2:
                    gpu_type = ET.SubElement(parent, "jsdl:GPUArchitecture")
                    gpu_type_name = ET.SubElement(gpu_type, "jsdl:GPUArchitectureName")
                    gpu_type_name.text = type_count[0]
            else:
                raise ValueError("Unsupported generic resource: %s" % resource)
    return parent


def job_scratch(parent):
    scratch = ET.SubElement(parent, "jsdl:FileSystem", name="SCRATCH")
    scratch.set("xmlns:ns10", "http://vcgr.cs.virginia.edu/genesisII/jsdl")
    scratch.set("ns10:unique-id", "test_scratch")
    fs_type = ET.SubElement(scratch, "jsdl:FileSystemType")
    fs_type.text = "spool"
    return parent


def job_stage_data(parent, name: Path, path: Path, is_output=True):
    el = ET.SubElement(parent, "jsdl:DataStaging")
    file_name = ET.SubElement(el, "jsdl:FileName")
    file_name.text = str(name)
    if is_output:
        file_path = ET.SubElement(el, "jsdl:Target")
    else:
        file_path = ET.SubElement(el, "jsdl:Source")
    path_uri = ET.SubElement(file_path, "jsdl:URI")
    path_uri.text = str(path)
    creation_flag = ET.SubElement(el, "jsdl:CreationFlag")
    delete_on_term = ET.SubElement(el, "jsdl:DeleteOnTermination")
    handle_as_archive = ET.SubElement(el, "jsdl:HandleAsArchive")
    if ".tar" in name.suffixes or ".zip" in name.suffixes:
        handle_as_archive.text = "true"
        # delete_on_term.text = "false"
        delete_on_term.text = "true"
        # creation_flag.text = "dontOverwrite"
        creation_flag.text = "overwrite"
        # fs = ET.SubElement(el, "jsdl:FilesystemName")
        # fs.text = "SCRATCH"
    else:
        handle_as_archive.text = "false"
        delete_on_term.text = "true"
        creation_flag.text = "overwrite"
    always_stage_out = ET.SubElement(el, "jsdl:AlwaysStageOut")
    always_stage_out.text = "false"
    return el


def prettystring(xml):
    xml_string = ET.tostring(xml)
    reparsed_xml = minidom.parseString(xml_string)
    return reparsed_xml.toprettyxml(indent="    ")


def prettyprint(xml):
    print(prettystring(xml))


def main(args: argparse.Namespace) -> None:
    stdout_path = args.grid_path / args.output
    stderr_path = args.grid_path / args.error
    job_def = job_definition()
    job_desc = job_description(job_def)
    job_id = job_identification(job_desc, args.job_name)
    job_app = job_application(
        job_desc,
        executable=args.executable,
        args=args.args,
        output=stdout_path,
        error=stderr_path,
    )
    job_res = job_resources(job_desc)
    job_res = job_time(job_res, args.time)
    job_res = job_memory(job_res, args.mem)
    job_res = job_gres(job_res, args.gres)
    job_res = job_scratch(job_res)

    job_data = []
    for file_name in args.input_data:
        job_data.append(
            job_stage_data(
                job_desc, file_name, args.grid_path / file_name, is_output=False
            )
        )
    for file_name in args.output_data:
        job_data.append(job_stage_data(job_desc, file_name, args.grid_path / file_name))
    job_data.append(job_stage_data(job_desc, args.output, stdout_path))
    job_data.append(job_stage_data(job_desc, args.error, stderr_path))

    prettyprint(job_def)

    job_fd, job_file_name = tempfile.mkstemp()
    with open(job_fd, "w") as job_file:
        job_file.write(prettystring(job_def))
    sp.run(
        shlex.split(
            "grid qsub /resources/CCC/Lancium/queues/RegularQueue local:%s"
            % job_file_name
        )
    )
    os.remove(job_file_name)


if __name__ == "__main__":
    main(_parse_args())
