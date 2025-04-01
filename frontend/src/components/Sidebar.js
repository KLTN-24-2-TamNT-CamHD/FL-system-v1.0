import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import { FaUsers, FaPlay, FaChartLine, FaLink, FaBuilding, FaCloudUploadAlt } from 'react-icons/fa'; // Import the Cloud icon

const SidebarContainer = styled.div`
  width: 200px;
  background-color: #333;
  color: white;
  padding: 20px 0;
  height: 100vh;
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItem = styled.li`
  padding: 10px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;

  a {
    color: white;
    text-decoration: none;
    display: flex;
    align-items: center;
  }

  svg {
    margin-right: 10px;
  }

  &:hover {
    background-color: #555;
  }
`;

const Sidebar = () => {
    return (
        <SidebarContainer>
            <NavList>
                <NavItem><Link to="/clients"><FaUsers />Clients</Link></NavItem>
                <NavItem><Link to="/training"><FaPlay />Training</Link></NavItem>
                <NavItem><Link to="/fl-training"><FaCloudUploadAlt />FL Training</Link></NavItem>
                <NavItem><Link to="/model"><FaChartLine />Model Monitor</Link></NavItem>
                <NavItem><Link to="/blockchain"><FaLink />Blockchain</Link></NavItem>
                <NavItem><Link to="/register"><FaBuilding />Register Institution</Link></NavItem>
            </NavList>
        </SidebarContainer>
    );
};

export default Sidebar;