import base64
import csv
import email.message
import hashlib


def create_dist_info_dir(container, name, version):
    print("create_dist_info_dir()")
    dist_info = container / f"{name}-{version}.dist-info"
    dist_info.mkdir()
    return dist_info


def write_metadata(dist_info, name, version):
    print("write_metadata()")
    m = email.message.EmailMessage()  # RFC 822.
    m["Metadata-Version"] = "2.1"
    m["Name"] = name
    m["Version"] = version
    dist_info.joinpath("METADATA").write_bytes(bytes(m))


def _record_row_from_path(path, relative):
    file_data = path.read_bytes()
    file_size = len(file_data)
    file_hash = (
        base64.urlsafe_b64encode(hashlib.sha256(file_data).digest())
        .decode("latin1")
        .rstrip("=")
    )
    return [relative.as_posix(), f"sha256={file_hash}", str(file_size)]


def iter_files(roots):
    for root in roots:
        for path in root.glob("**/*"):
            if not path.is_file():
                continue
            if path.suffix == ".pyc" or path.parent.name == "__pycache__":
                continue
            yield path, path.relative_to(root.parent)


def write_record(dist_info, package):
    print("write_record()")
    with dist_info.joinpath("RECORD").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        for path, relative in iter_files((package, dist_info)):
            w.writerow(_record_row_from_path(path, relative))
        w.writerow([f"{dist_info.name}/RECORD", "", ""])
