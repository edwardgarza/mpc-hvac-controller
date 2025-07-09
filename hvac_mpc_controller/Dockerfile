FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

USER root

# Install cron for Alpine Linux
RUN apk add --no-cache dcron

# Create cron job file
RUN echo "*/15 * * * * cd / && python3 generate_config.py && python3 automation_runner.py >> /var/log/cron.log 2>&1" > /etc/crontabs/root
RUN chmod 0644 /etc/crontabs/root

# Create log file
RUN touch /var/log/cron.log

# Copy our updated run script that includes cron
COPY run.sh /
RUN chmod a+x /run.sh

USER app

# Use our run script that starts both cron and the server
CMD ["/run.sh"] 