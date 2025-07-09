FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

# Switch to root temporarily to copy files and set permissions
USER root

COPY run.sh /
COPY generate_config.py /

# Debug: check what files are present
RUN ls -la /

# Make run.sh executable
RUN chmod +x /run.sh

# Switch back to non-root user
USER app

CMD ["/run.sh"] 