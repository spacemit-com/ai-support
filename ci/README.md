## CI Variables

| Variable Name | Default Value | Description |
| :--: | :--: | :--: |
| CI_PKG_OPTION | --skip-py | options during ci `package` stage, choose values from [--skip-deb \| --skip-ort \| --skip-py \| --debug] |

## Bianbu DevScripts

* step by step

```bash
# preparation: download bianbu development toolkits
git clone -b main git@gitlab.dc.com:bianbu/bianbu-devscripts.git && cd bianbu-devscripts
# preparation: install dependencies
apt update && apt install sudo vim-common -y
bash install-scripts-depends.sh
# preparation: install ca-certificates
apt-get -y install ca-certificates wget && wget https://nexus.bianbu.xyz/repository/software/dc.com-CA-Root-Base64.crt -O /usr/local/share/ca-certificates/dc.com-CA-Root-Base64.crt --no-check-certificate && update-ca-certificates
# preparation: add bianbu source list
bianbu-dev chroot
# add mantic-spacemit(similar for mantic-porting, mantic-customization)
cat <<EOF >>/etc/apt/sources.list.d/bianbu.list
# mantic-spacemit
deb [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-spacemit main universe multiverse restricted
deb-src [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-spacemit main universe multiverse restricted
EOF
apt update
exit # quit chroot

# quick start: check toolkit version
PATH=$(pwd):$PATH
bianbu-dev -v
bianbu-pkg -h
# quick start: default settings
bianbu-dev set-default-dist bianbu-23.10
bianbu-dev set-default-env bianbu-scripts-env-1.8.1
# quick start: install docker container for development(optional)
bianbu-dev install

# -------------------------------------------------------------

# quick import: example with gnome-shell
# Note: To import-dsc, plz switch to any non-git repository directory
apt source --download-only gnome-shell # gnome-shell_*-0ubuntu1.debian.tar.xz, gnome-shell_*-0ubuntu1.dsc, gnome-shell_*.orig.tar.xz
bianbu-dev import-dsc gnome-shell_*-0ubuntu1.dsc
# quick glance
tree gnome-shell -L 1

# quick import: with bianbu-ai-support
cd bianbu-ai-support && git checkout main
# Note: version number must start with digit(required by dpkg-buildpackage)
bianbu-dev import . --suite mantic-porting --new-version 1.0.1 # change from spacemit to porting from v1.0.13

# -------------------------------------------------------------

# do sth ...
git checkout -b bianbu-23.10 # switch to bianbu-23.10
bianbu-dev tag
bianbu-dev pack -a  # same as `bianbu-dev pack && bianbu-dev build` (pack 1st, then build)

# -------------------------------------------------------------

# enter develop env
bianbu-dev chroot   # apt install onnxruntime, libopencv-dev

# build package with dpkg(Note: install dependencies 1st plz, e.g. onnxruntime, opencv)
apt-get build-dep -y $(pwd) # run in bianbu-ai-support directory
dpkg-buildpackage -us -b -uc --no-pre-clean --no-post-clean # -tc
# ..
# |-- bianbu-ai-support-dbgsym_1.0.1_amd64.ddeb
# |-- bianbu-ai-support_1.0.1_amd64.buildinfo
# |-- bianbu-ai-support_1.0.1_amd64.changes
# |-- bianbu-ai-support_1.0.1_amd64.deb

# -------------------------------------------------------------

# check https://archive.bianbu.xyz/bianbu-ports/logs/ for upload events
bianbu-dev upload onnxruntime_1.15.1_amd64.deb --suite mantic-porting
bianbu-dev upload onnxruntime_1.15.1_riscv64.deb --suite mantic-porting

apt update
apt show onnxruntime
apt install onnxruntime
```