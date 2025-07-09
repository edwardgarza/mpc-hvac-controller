FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

# Switch to root to copy files and set permissions
USER root

COPY run.sh /
RUN chmod a+x /run.sh

COPY generate_config.py /

CMD ["/run.sh"] 