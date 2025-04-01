import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table } from '../styles';

const ClientList = () => {
    const [clients, setClients] = useState([]);

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/clients') // Replace with your FastAPI endpoint
            .then(response => setClients(response.data))
            .catch(error => console.error('Error fetching clients:', error));
    }, []);

    return (
        <div>
            <h2>Client List</h2>
            <Table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        {/* Add more columns as needed */}
                    </tr>
                </thead>
                <tbody>
                    {clients.map(client => (
                        <tr key={client.id}>
                            <td>{client.id}</td>
                            <td>{client.name}</td>
                            {/* Add more cells as needed */}
                        </tr>
                    ))}
                </tbody>
            </Table>
        </div>
    );
};

export default ClientList;