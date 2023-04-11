import email.message
import zipfile

from .distinfo import create_dist_info_dir, iter_files, write_metadata, write_record


def write_wheel_metadata(dist_info, tag):
    print("write_wheel_metadata()")
    m = email.message.EmailMessage()
    m["Wheel-Version"] = "1.0"
    m["Generator"] = "packager/wheel.py"
    m["Root-Is-Purelib"] = "true"
    m["Tag"] = tag
    dist_info.joinpath("WHEEL").write_bytes(bytes(m))


def create_dist_info(name, version, tag, package, output_dir):
    print("create_dist_info()")
    dist_info = create_dist_info_dir(output_dir, name, version)
    write_metadata(dist_info, name, version)
    write_wheel_metadata(dist_info, tag)
    write_record(dist_info, package)
    return dist_info


def create_wheel(name, version, tag, package, *, dist_info, libxgboost, output_dir):
    print("create_wheel()")
    wheel_path = output_dir / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        for path, relative in iter_files((package, dist_info)):
            zf.write(path, relative.as_posix())
        zf.write(libxgboost, f"xgboost/lib/{libxgboost.name}")
    return wheel_path
