## Auto Construction For Debian

* create new container

```bash
IMAGE=harbor.bianbu.xyz/dev/bianbu-scripts-env:1.8.1
NAME=xxx
docker run --restart=always --net=host --privileged -itd --name $NAME -v $HOME:/mnt $IMAGE /bin/bash
```

* enter container

```bash
docker exec -it $NAME schroot -r -c persistent-session
```

* update container

```bash
apt update
apt install -y cmake libopencv-dev
 
cat <<EOF >>/etc/apt/sources.list.d/bianbu.list
# mantic-spacemit
deb [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-spacemit main universe multiverse restricted
deb-src [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-spacemit main universe multiverse restricted
# mantic-porting
deb [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-porting main universe multiverse restricted
deb-src [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-porting main universe multiverse restricted
# mantic-customization
#deb [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-customization main universe multiverse restricted
#deb-src [signed-by=/usr/share/keyrings/bianbu-archive-keyring.gpg] https://archive.bianbu.xyz/bianbu-ports/ mantic-customization main universe multiverse restricted
EOF

apt update
apt install -y debhelper onnxruntime

# update /opt/smartentry/HEAD/pre-run* for smartentry(/sbin/smartentry.sh)
vi /opt/smartentry/HEAD/pre-run
# ...
```

* save/export container

```bash
docker commit $NAME <new-image>
```

* /opt/smartentry/HEAD/pre-run

```bash
# 此脚本会在docker run 时执行
# 可以在这里做一些修改，补丁。执行完才会执行run脚本中的内容
if [ -d "/var/lib/schroot/unpack/persistent-session" ]; then
    echo "persistent-session exists."
    schroot --recover-session -c persistent-session
else
    echo "persistent-session does not exist."
    # start a new session with specified chroot named as persistent-session
    schroot -b -c mantic-riscv64-sbuild -n persistent-session

fi
echo 'docker inited' > /root/.smartentry_status
mount -o bind /mnt /var/run/schroot/mount/persistent-session/mnt
```

* smoke gitlab ci test

```bash
# run specified session
cp -rdf $CI_PROJECT_DIR/.. /mnt/
schroot -r -c persistent-session --directory /mnt/ai/support
apt build-dep -y $(pwd)
dpkg-buildpackage -us -b -uc --no-pre-clean --no-post-clean
exit
```