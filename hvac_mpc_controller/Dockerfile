FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

USER root

# Install cron for Debian/Ubuntu
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy our updated run script
COPY run.sh /
RUN chmod a+x /run.sh

# Use our run script that starts both cron and the server
CMD ["/run.sh"] 