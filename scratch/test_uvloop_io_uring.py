import uvloop
import asyncio
import os

async def main():
    print("Loop running")
    # Check if there are any io_uring FDs for this process
    pid = os.getpid()
    fds = os.listdir(f"/proc/{pid}/fd")
    found = False
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
            if "io_uring" in target:
                print(f"Found io_uring FD: {fd} -> {target}")
                found = True
        except:
            pass
    if not found:
        print("No io_uring FDs found")

if __name__ == "__main__":
    print(f"UV_USE_IO_URING={os.environ.get('UV_USE_IO_URING')}")
    uvloop.run(main())
