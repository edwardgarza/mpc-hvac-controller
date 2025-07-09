FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

USER root

# Install cron for Debian/Ubuntu
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Create cron job file
RUN echo "*/15 * * * * cd / && python3 generate_config.py && python3 automation_runner.py >> /var/log/cron.log 2>&1" > /etc/cron.d/hvac-automation
RUN chmod 0644 /etc/cron.d/hvac-automation

# Create log file
RUN touch /var/log/cron.log

# Copy our updated run script that includes cron
COPY run.sh /
RUN chmod a+x /run.sh

# Use our run script that starts both cron and the server
CMD ["/run.sh"] 